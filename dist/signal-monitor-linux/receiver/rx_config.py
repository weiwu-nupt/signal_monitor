"""
接收机配置持久化
=================
对应文档中的 MRConfig.ini.
默认值取自配置文档第 2 节.
"""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass, asdict


CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "MRConfig.ini")


@dataclass
class RxConfig:
    receiver_ip: str = "192.168.0.138"   # 默认接收机IP
    local_ip: str = "192.168.0.120"      # 默认本地IP
    tcp_port: int = 5025                 # TCP 控制端口固定 5025

    @classmethod
    def load(cls, path: str = CONFIG_PATH) -> "RxConfig":
        cfg = cls()
        if not os.path.exists(path):
            return cfg
        cp = configparser.ConfigParser()
        try:
            cp.read(path, encoding="utf-8")
            if cp.has_section("network"):
                s = cp["network"]
                cfg.receiver_ip = s.get("receiver_ip", cfg.receiver_ip)
                cfg.local_ip = s.get("local_ip", cfg.local_ip)
                cfg.tcp_port = s.getint("tcp_port", cfg.tcp_port)
        except (configparser.Error, ValueError):
            pass
        return cfg

    def save(self, path: str = CONFIG_PATH) -> None:
        cp = configparser.ConfigParser()
        cp["network"] = {
            "receiver_ip": self.receiver_ip,
            "local_ip": self.local_ip,
            "tcp_port": str(self.tcp_port),
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                cp.write(f)
        except OSError:
            pass

    def as_dict(self) -> dict:
        return asdict(self)