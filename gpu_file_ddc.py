"""GPU file simulation of a concurrent FPGA DDC batch; no IQ padding."""
import numpy as np
import torch


@torch.inference_mode()
def collect_gpu_batch(info, start, channels, ready, cancelled, progress, accepts, device):
    from fpga_closed_loop import IQ_BYTES
    device = torch.device(device)
    wanted = IQ_BYTES//4
    stride, guard = round(info.sample_rate*.01), round(info.sample_rate*.005)
    count = stride+2*guard
    spacing = info.sample_rate/count
    logical_start = start-guard
    stream = torch.cuda.Stream(device=device)
    groups = {}
    # Group equal output rates into batched inverse FFTs. Cap temporary
    # frequency matrices at about two million bins per group.
    for target, identifier in sorted(channels, key=lambda lane: -lane[0].sample_rate_hz):
        if cancelled():
            return
        if not accepts(target):
            continue
        rate = int(target.sample_rate_hz)
        if not 3*target.bandwidth_hz <= rate <= min(4*target.bandwidth_hz, info.sample_rate):
            raise ValueError('IQ输出采样率必须为带宽3～4倍且不超过输入采样率')
        width = round(rate/spacing)
        if width < 2 or width > count or width % 2 or abs(width*spacing-rate) > 1e-6:
            raise ValueError('DDC块时长与输出采样率不匹配')
        entries = groups.setdefault(rate, [])
        center_bin = round(target.center_hz/spacing)
        residual = target.center_hz-center_bin*spacing
        offset = (np.arange(width)-width//2)*spacing-residual
        response = .5*(1+np.cos(np.pi*np.clip(
            (np.abs(offset)-target.bandwidth_hz/2)/(target.bandwidth_hz*.125), 0, 1)))
        physical = target.center_hz+offset
        response[(physical < -info.sample_rate/2) | (physical >= info.sample_rate/2)] = 0
        indices = (np.arange(width)-width//2+center_bin) % count
        mixer = np.exp(-2j*np.pi*residual*np.arange(width)/rate).astype(np.complex64)
        entries.append((target, identifier, indices, response.astype(np.float32), mixer))

    plans = []
    with torch.cuda.stream(stream):
        for rate, entries in groups.items():
            width = len(entries[0][2])
            limit = max(1, 2_000_000//width)
            for offset in range(0, len(entries), limit):
                part = entries[offset:offset+limit]
                plans.append(dict(rate=rate, width=width, filled=0,
                    lanes=[(t, i, torch.empty((wanted, 2), dtype=torch.int16, device=device))
                           for t, i, *_ in part],
                    indices=torch.from_numpy(np.stack([p[2] for p in part])).to(device),
                    response=torch.from_numpy(np.stack([p[3] for p in part])).to(device),
                    mixer=torch.from_numpy(np.stack([p[4] for p in part])).to(device)))
        groups.clear()  # Release CPU plans after upload.
        with info.path.open('rb') as file:
            while plans:
                if cancelled():
                    return
                remaining, position = count, logical_start % info.samples
                pieces = []
                while remaining:
                    if cancelled():
                        return
                    take = min(remaining, info.samples-position)
                    file.seek(info.offset+position*4)
                    data = file.read(take*4)
                    if len(data) != take*4:
                        raise ValueError('DDC采集期间文件被截断')
                    pieces.append(data)
                    remaining -= take
                    position = 0
                raw = np.frombuffer(b''.join(pieces), dtype='<i2').reshape(-1, 2)
                components = torch.from_numpy(raw.copy()).to(device).float()
                fft = torch.fft.fft(torch.complex(components[:, 0], components[:, 1]))
                pending = []
                for plan in plans:
                    if cancelled():
                        return
                    selected = [i for i, lane in enumerate(plan['lanes']) if accepts(lane[0])]
                    if not selected:
                        continue
                    if len(selected) != len(plan['lanes']):
                        selection = torch.tensor(selected, device=device)
                        plan['lanes'] = [plan['lanes'][i] for i in selected]
                        for name in ('indices', 'response', 'mixer'):
                            plan[name] = plan[name].index_select(0, selection)
                    width, rate = plan['width'], plan['rate']
                    bins = fft[plan['indices']]*plan['response']
                    channel = torch.fft.ifft(torch.fft.ifftshift(bins, dim=1), dim=1)
                    phases = np.array([(t.center_hz*logical_start/info.sample_rate) % 1.0
                                       for t, _, _ in plan['lanes']])
                    phase = torch.from_numpy(np.exp(-2j*np.pi*phases).astype(np.complex64)).to(device)
                    channel *= (width/count)*plan['mixer']*phase[:, None]
                    trim = round(rate*.005)
                    middle = channel[:, trim:width-trim]
                    take = min(middle.shape[1], wanted-plan['filled'])
                    iq = torch.stack((middle[:, :take].real, middle[:, :take].imag), dim=2)
                    iq = iq.round().clamp(-32768, 32767).to(torch.int16)
                    filled = plan['filled']
                    for index, (_, identifier, output) in enumerate(plan['lanes']):
                        output[filled:filled+take].copy_(iq[index])
                        if progress:
                            progress(identifier, filled+take, wanted)
                    plan['filled'] += take
                    if plan['filled'] == wanted:
                        # The consumer uses its own CUDA stream. Publish only
                        # after producer writes are complete; retain IQ on GPU.
                        stream.synchronize()
                        for target, identifier, output in plan['lanes']:
                            ready(target, identifier, output, rate)
                    else:
                        pending.append(plan)
                plans = pending
                logical_start += stride
                stream.synchronize()
