"""
Depo Teslimat Robotu - Otonom Navigasyon Simülasyonu
Mobil Robotlar Ödevi

Kullanılan yapay zeka: Claude (claude-sonnet-4-6) / GitHub Copilot
Kullanılan bölümler: Kod iskeleti, algoritma implementasyonu, arayüz tasarımı
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.widgets import RadioButtons, Button, Slider
import heapq, random, math, time

# ─── RENKLER  (Claude açık tema) ─────────────────────────────────────
DARK  = "#f5efe6"   # ana arka plan — sıcak krem
PANEL = "#ede4d8"   # panel arka planı — bej
ACC   = "#c85f35"   # Claude turuncu (koyu yüzey için daha derin)
# Metin
C_BRIGHT = "#2d1f0f"  # ana metin — koyu sıcak kahve
C_MID    = "#7a5c3a"  # ikincil metin — orta kahve
C_DIM    = "#b09878"  # soluk metin — açık kahve
# Durum
C_OK   = "#4a8c30"    # yeşil
C_WARN = "#b86820"    # amber
C_ERR  = "#b83820"    # kırmızı-turuncu
# Veri çizgileri — açıkça ayırt edilebilir üç renk
C_TRUE = "#2a5298"    # koyu mavi  (gerçek rota)
C_EKF  = "#c85f35"    # Claude turuncu (EKF tahmini)
C_DR   = "#2a8a48"    # orman yeşili (Dead Reckoning)
# Harita arkaplanı ve kenarlıklar
MAP_BG  = "#faf7f2"   # harita içi — çok açık krem
SPINE_C = "#c8b090"   # eksen kenarı
GRID_C  = "#e0d4c0"   # ızgara
OBSTACLE_COLOR = "#3d2010"   # koyu kahve — tüm engeller


def _darken(hex_col, amount=0.3):
    hex_col = hex_col.lstrip("#")
    r, g, b = (int(hex_col[i:i+2], 16) for i in (0, 2, 4))
    r = int(r * (1 - amount))
    g = int(g * (1 - amount))
    b = int(b * (1 - amount))
    return f"#{r:02x}{g:02x}{b:02x}"


# ... TÜM DOSYA İÇERİĞİ YUKARIDAKİ GBİ DEVAM EDİYOR ...

# ─── GİRİŞ ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    gui=GUI()
    gui.run()
