"""
Linux端频谱读取程序：读取FPGA写到DDR的频谱packet，并自动ACK释放slot。

启动FPGA回放后，直接运行(单次采集)：
    sudo python3 spectrum_reader.py

显示端集成时，只需要复用read_one_spectrum()：
    result = reader.read_one_spectrum()
    if result is not None:
        power_db = result["power_db"]
        # update_plot(power_db)

当前FPGA协议：
    status地址 : 0x00000380，FPGA写，Linux读（观察信号用的，不必理会）
    ACK地址    : 0x000003C0，Linux写，FPGA读（清除标志位，释放已经读完的slot）
    slot0地址  : 0x10000000
    slot1地址  : 0x10020000(备用的，不用理会)
    每包长度    : 65664字节 = 64B包头 + 8192*8B raw power + 64B包尾

slot切换方式：
    - Linux不需要手动指定slot；
    - 程序会自动读status；
    - 哪个slot valid就读哪个；
    - 两个slot都valid时，优先读seq更小的旧包；
    - 读完并校验通过后，自动写ACK；
    - FPGA收到ACK后释放该slot。

显示端实际上直接获取结果即可

result = read_one_spectrum(c2h_fd, h2c_fd, allowed_sync_words) 获取结果
power_db = result["power_db"]

"""

import os
import time
import struct
import argparse
import numpy as np


# ============================================================
# 1. XDMA设备路径
# ============================================================
C2H_DEV = "/dev/xdma0_c2h_0"   # FPGA DDR 到 Linux
H2C_DEV = "/dev/xdma0_h2c_0"   # Linux 到 FPGA DDR，用于写 ACK


# ============================================================
# 2. FPGA端地址
# ============================================================
DISP_STATUS_ADDR = 0x00000380
DISP_ACK_ADDR    = 0x000003C0

SLOT0_BASE = 0x10000000
SLOT1_BASE = 0x10020000

PACKET_READ_BYTES = 65664


# ============================================================
# 3. 协议常量
# ============================================================
DSPSTAT_MAGIC = 0x4453505354415431  # "DSPSTAT1"
DSPACK_MAGIC  = 0x44535041434B3031  # "DSPACK01"

#包头，1A2B3C4D用作验证这个可以不用管，1ACFFC1D用于实际
DEFAULT_ALLOWED_SYNC_WORDS = (0x1A2B3C4D, 0x1ACFFC1D)

MSG_TYPE = 0x01
MSG_LEN  = 65544

CENTER_MHZ = 0
BW_MHZ     = 410
FFT_N      = 8192


# ============================================================
# 4. XDMA底层读写函数，直接调用即可
# ============================================================
def read_at(fd: int, addr: int, size: int) -> bytes:
    """从 XDMA C2H 设备指定地址读取 size 字节。"""
    os.lseek(fd, addr, os.SEEK_SET)

    data = bytearray()
    while len(data) < size:
        chunk = os.read(fd, size - len(data))
        if not chunk:
            raise RuntimeError("XDMA 读取失败：返回空数据")
        data.extend(chunk)

    return bytes(data)

def write_at(fd: int, addr: int, data: bytes):
    """向 XDMA H2C 设备指定地址写入 data。"""
    os.lseek(fd, addr, os.SEEK_SET)

    offset = 0
    while offset < len(data):
        n = os.write(fd, data[offset:])
        if n <= 0:
            raise RuntimeError("XDMA 写入失败")
        offset += n


# ============================================================
# 5. CRC16 计算这一块用来在Linux端重新计算CRC，然后和FPGA包尾里的CRC对比。
# ============================================================
def crc16_ccitt_false(data: bytes, init: int = 0xFFFF) -> int:
    """
    CRC16-CCITT-FALSE。

    FPGA 当前 tail[15:0] 的 CRC16 只覆盖 6 个字节：
        header[0x09:0x0F]
        = center_freq[2B] + bandwidth[2B] + fft_points[2B]
    """
    crc = init
    for b in data:
        crc ^= (b << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) & 0xFFFF) ^ 0x1021
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF


# ============================================================
# 6. 解析 FPGA display status 这个用于读取板卡状态的，用于观察信号
# ============================================================
def parse_status(buf: bytes) -> dict:
    """
    display status 布局，地址 0x00000380，长度 64B：
        +0x00 uint64 magic
        +0x08 uint64 flags
              bit2 = slot0_valid
              bit3 = slot1_valid
        +0x18 uint32 slot0_base
        +0x1C uint32 slot1_base
        +0x20 uint32 slot0_seq
        +0x24 uint32 slot1_seq
        +0x28 uint64 drop_count
        +0x30 uint32 next_seq
        +0x38 uint32 msg_len
        +0x3C uint32 read_bytes
    """
    if len(buf) < 64:
        raise ValueError("status 长度不足 64 字节")

    magic = struct.unpack_from("<Q", buf, 0x00)[0]
    flags = struct.unpack_from("<Q", buf, 0x08)[0]

    return {
        "magic": magic,
        "flags": flags,
        "slot0_valid": (flags >> 2) & 1,
        "slot1_valid": (flags >> 3) & 1,
        "slot0_base": struct.unpack_from("<I", buf, 0x18)[0],
        "slot1_base": struct.unpack_from("<I", buf, 0x1C)[0],
        "slot0_seq":  struct.unpack_from("<I", buf, 0x20)[0],
        "slot1_seq":  struct.unpack_from("<I", buf, 0x24)[0],
        "drop_count": struct.unpack_from("<Q", buf, 0x28)[0],
        "next_seq":   struct.unpack_from("<I", buf, 0x30)[0],
        "msg_len":    struct.unpack_from("<I", buf, 0x38)[0],
        "read_bytes": struct.unpack_from("<I", buf, 0x3C)[0],
    }


# ============================================================
# 7. 自动选择slot
# ============================================================
def choose_slot(st: dict):
    """
    自动选择要读取的 slot。

    规则：
        1. slot0 和 slot1 都有效：读 seq 更小的旧包；
        2. 只有 slot0 有效：读 slot0；
        3. 只有 slot1 有效：读 slot1；
        4. 都无效：返回None。
    """
    if st["slot0_valid"] and st["slot1_valid"]:
        if st["slot0_seq"] <= st["slot1_seq"]:
            return 0, st["slot0_base"] or SLOT0_BASE, st["slot0_seq"]
        return 1, st["slot1_base"] or SLOT1_BASE, st["slot1_seq"]

    if st["slot0_valid"]:
        return 0, st["slot0_base"] or SLOT0_BASE, st["slot0_seq"]

    if st["slot1_valid"]:
        return 1, st["slot1_base"] or SLOT1_BASE, st["slot1_seq"]

    return None


# ============================================================
# 8. 解析packet
# ============================================================
def parse_packet(buf: bytes, expected_slot: int, expected_seq: int, allowed_sync_words) -> dict:
    """
    packet 布局：
        +0x00000 ~ +0x0003F : 64B header
        +0x00040 ~ +0x1003F : 8192 个 uint64 raw power
        +0x10040 ~ +0x1007F : 64B tail

    header：
        byte[0:3]   sync word  
        byte[4]     msg_type
        byte[5:8]   msg_len
        byte[9:10]  center MHz
        byte[11:12] bandwidth MHz
        byte[13:14] fft points
        byte[15]    slot_id
        byte[16:19] seq_id

    tail：
        byte[0:1]   CRC16
        byte[2:5]   seq_id
        byte[6]     slot_id
    """
    if len(buf) < PACKET_READ_BYTES:
        raise ValueError("packet长度不足")

    sync     = struct.unpack_from("<I", buf, 0x00)[0]  #帧头
    msg_type = buf[0x04]                               #类型
    msg_len  = struct.unpack_from("<I", buf, 0x05)[0]  #长度
    center   = struct.unpack_from("<H", buf, 0x09)[0]  #中心频率
    bw       = struct.unpack_from("<H", buf, 0x0B)[0]  #带宽
    fft_n    = struct.unpack_from("<H", buf, 0x0D)[0]  #fft点数
    slot_id  = buf[0x0F] & 0x1                         #slot区
    seq_hdr  = struct.unpack_from("<I", buf, 0x10)[0]  #连续性

    power = np.frombuffer(buf, dtype="<u8", count=FFT_N, offset=0x40).copy()    #数据的主要部分

    crc_rx    = struct.unpack_from("<H", buf, 0x10040)[0]
    seq_tail  = struct.unpack_from("<I", buf, 0x10042)[0]
    slot_tail = buf[0x10046] & 0x1
    crc_calc  = crc16_ccitt_false(buf[0x09:0x0F])

    checks = {
        "sync_ok":      sync in allowed_sync_words,
        "type_ok":      msg_type == MSG_TYPE,
        "msg_len_ok":   msg_len == MSG_LEN,
        "center_ok":    center == CENTER_MHZ,
        "bw_ok":        bw == BW_MHZ,
        "fft_n_ok":     fft_n == FFT_N,
        "slot_hdr_ok":  slot_id == expected_slot,
        "seq_hdr_ok":   seq_hdr == expected_seq,
        "crc_ok":       crc_rx == crc_calc,
        "seq_tail_ok":  seq_tail == expected_seq,
        "slot_tail_ok": slot_tail == expected_slot,
    }

    return {
        "packet_ok": all(checks.values()),
        "checks": checks,
        "sync": sync,
        "msg_type": msg_type,
        "msg_len": msg_len,
        "center": center,
        "bw": bw,
        "fft_n": fft_n,
        "slot_id": slot_id,
        "seq_hdr": seq_hdr,
        "crc_rx": crc_rx,
        "crc_calc": crc_calc,
        "seq_tail": seq_tail,
        "slot_tail": slot_tail,
        "power": power,
    }


# ============================================================
# 9. 生成ACK：通知FPGA释放已经读完的slot
# ============================================================
def make_ack(slot_id: int, seq: int) -> bytes:
    """
    生成 64 字节 ACK 数据。

    Linux 读完某个 slot 的频谱 packet，并且校验通过后，
    把 ACK 写到 FPGA 的 ACK 地址 0x000003C0。

    ACK 布局：
        +0x00 uint64 magic = DSPACK01
        +0x08 uint64 ack_bits
              bit0 = 1 表示请求释放 slot0
              bit1 = 1 表示请求释放 slot1
        +0x10 uint64 ack_seq0
        +0x18 uint64 ack_seq1

    FPGA 只有在 ACK 里的 seq 和当前 slot_seq 完全一致时，
    才会真正释放该 slot。
    """
    buf = bytearray(64)

    ack_bits = 1 << slot_id

    struct.pack_into("<Q", buf, 0x00, DSPACK_MAGIC)
    struct.pack_into("<Q", buf, 0x08, ack_bits)

    if slot_id == 0:
        struct.pack_into("<Q", buf, 0x10, seq)
        struct.pack_into("<Q", buf, 0x18, 0)
    else:
        struct.pack_into("<Q", buf, 0x10, 0)
        struct.pack_into("<Q", buf, 0x18, seq)

    return bytes(buf)


# ============================================================
# 10. 读取一帧频谱
# ============================================================
def read_one_spectrum(c2h_fd: int, h2c_fd: int, allowed_sync_words):
    """
    读取一帧频谱并自动 ACK。

    返回：
        None：当前没有新 packet；
        dict：包含包头信息、包尾信息、raw power。
    """
    status_buf = read_at(c2h_fd, DISP_STATUS_ADDR, 64)   #读 0x00000380 status
    st = parse_status(status_buf)                        #parse_status()解析status

    if st["magic"] != DSPSTAT_MAGIC:                     #检查status magic是否正确
        raise RuntimeError("display status magic 错误，请检查FPGA是否已经运行或地址是否一致")

    chosen = choose_slot(st)                             #choose_slot() 自动选择slot0或slot1
    if chosen is None:
        return None                                       #choose_slot() 自动选择slot0或slot1

    slot_id, base, seq = chosen
    read_bytes = st["read_bytes"] or PACKET_READ_BYTES

    if read_bytes != PACKET_READ_BYTES:
        raise RuntimeError("packet 长度异常：read_bytes=%d，期望=%d" % (read_bytes, PACKET_READ_BYTES))

    packet_buf = read_at(c2h_fd, base, read_bytes)          #根据slot base读取完整packet

    pkt = parse_packet(                                   #parse_packet() 解析包头、power 数据、包尾和CRC
        packet_buf,
        expected_slot=slot_id,
        expected_seq=seq,
        allowed_sync_words=allowed_sync_words,
    )

    if not pkt["packet_ok"]:
        bad_items = [name for name, ok in pkt["checks"].items() if not ok]
        raise RuntimeError(
            "packet 校验失败：bad=%s sync=0x%08X type=0x%02X msg_len=%d "
            "center=%d bw=%d fft_n=%d crc_rx=0x%04X crc_calc=0x%04X "
            "slot=%d seq=%d hdr_slot=%d hdr_seq=%d tail_slot=%d tail_seq=%d"
            % (
                ",".join(bad_items),
                pkt["sync"],
                pkt["msg_type"],
                pkt["msg_len"],
                pkt["center"],
                pkt["bw"],
                pkt["fft_n"],
                pkt["crc_rx"],
                pkt["crc_calc"],
                slot_id,
                seq,
                pkt["slot_id"],
                pkt["seq_hdr"],
                pkt["slot_tail"],
                pkt["seq_tail"],
            )
        )

    write_at(h2c_fd, DISP_ACK_ADDR, make_ack(slot_id, seq))    #清除标志位 写ACK到0x000003C0

    power = pkt["power"]
    power_db = 10.0 * np.log10(np.maximum(power.astype(np.float64), 1.0)) #转成db

    return {
        "slot_id": slot_id,
        "seq": seq,
        "status": st,
        "packet": pkt,
        "power": power,
        "power_db": power_db,
    }


# ============================================================
# 11. 输出packet包信息
# ============================================================
def print_packet_info(result: dict):
    """只输出 packet 包里的协议信息。"""
    pkt = result["packet"]
    st  = result["status"]

    print(
        "packet_ok=%s  \n"
        "slot=%d seq=%d  \n"
        "sync=0x%08X type=0x%02X msg_len=%d  \n"
        "center=%dMHz bw=%dMHz fft_n=%d \n"
        "crc_rx=0x%04X crc_calc=0x%04X \n"
        "tail_seq=%d tail_slot=%d \n"
        "read_bytes=%d drop_count=%d next_seq=%d \n"
        % (
            pkt["packet_ok"],
            result["slot_id"],
            result["seq"],
            pkt["sync"],
            pkt["msg_type"],
            pkt["msg_len"],
            pkt["center"],
            pkt["bw"],
            pkt["fft_n"],
            pkt["crc_rx"],
            pkt["crc_calc"],
            pkt["seq_tail"],
            pkt["slot_tail"],
            st["read_bytes"],
            st["drop_count"],
            st["next_seq"],
        )
    )


# ============================================================
# 12. 命令行入口
# ============================================================
def parse_sync_arg(sync_str: str):
    """
    不传 --sync：
        默认兼容 0x1A2B3C4D 和 0x1ACFFC1D。

    传 --sync：
        只接受指定同步字。
    """
    if sync_str is None:
        return tuple(DEFAULT_ALLOWED_SYNC_WORDS)

    s = sync_str.strip().lower()
    if s.startswith("0x"):
        value = int(s, 16)
    else:
        value = int(s, 16)
    return (value,)


def main():
    parser = argparse.ArgumentParser(description="读取FPGA输出的8192点频谱包，并输出packet包信息")
    parser.add_argument("--loop", action="store_true", help="持续读取，适合显示程序")
    parser.add_argument("--interval", type=float, default=0.05, help="没有新包时的等待时间，单位秒")
    parser.add_argument("--sync", type=str, default=None, help="强制指定0x1ACFFC1D")
    args = parser.parse_args()

    allowed_sync_words = parse_sync_arg(args.sync)

    c2h_fd = os.open(C2H_DEV, os.O_RDONLY | os.O_SYNC)  #打开XDMA读通道
    h2c_fd = os.open(H2C_DEV, os.O_WRONLY | os.O_SYNC)  #打开XDMA写通道

    try:
        while True:
            result = read_one_spectrum(c2h_fd, h2c_fd, allowed_sync_words)  #反复执行读一帧可以做到持续读的效果

            if result is None:                      #如果 result is None：当前没有新packet单次模式就退出 如果是循环模式就sleep后继续等
                if not args.loop:
                    print("当前没有新频谱包")
                    break
                time.sleep(args.interval)
                continue

            print_packet_info(result) # 如果 result 有数据，就打印

            if not args.loop:
                break

            time.sleep(args.interval)

    finally:
        os.close(c2h_fd)
        os.close(h2c_fd)


if __name__ == "__main__":
    main()