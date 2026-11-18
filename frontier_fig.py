import numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":0.8,
    "axes.edgecolor":"#333333","axes.grid":True,"grid.color":"#DDDDDD",
    "grid.linewidth":0.6,"figure.dpi":300})
INK,C1,C2,C3,C4 = "#1F1F1F","#7B1E3A","#2F5D8A","#B08A3E","#3F7A4E"

z = np.load("frontier_pts.npz", allow_pickle=True)
S = json.load(open("frontier_stats.json"))
risky = list(z["risky"])
mu, sd = z["mu"], z["sd"]
rf = float(z["rf"])

fig, ax = plt.subplots(figsize=(6.6, 4.6))

# frontiers (convert to %)
ax.plot(z["sig_n"]*100, z["ret_n"]*100, color="#888888", lw=1.8, ls="--",
        label="Frontier without wine (6 assets)")
ax.plot(z["sig_w"]*100, z["ret_w"]*100, color=C1, lw=2.4,
        label="Frontier with wine (7 assets)")

# individual assets
for t, x, y in zip(risky, sd, mu):
    col = C1 if t == "WINE" else C3
    ax.scatter(x*100, y*100, s=48, color=col, zorder=6, edgecolor="white", linewidth=0.8)
    dx, dy = (7, 5) if t not in ("TLT","WINE") else (7,-10)
    if t == "SPY": dx, dy = (-8, 8)
    ax.annotate(t if t!="WINE" else "Fine wine", (x*100, y*100),
                textcoords="offset points", xytext=(dx, dy), fontsize=8.5,
                fontweight="bold" if t=="WINE" else "normal",
                color=C1 if t=="WINE" else "#333333")

# global minimum variance points
mv_n = z["mv_n"]; mv_w = z["mv_w"]
ax.scatter(mv_n[0]*100, mv_n[1]*100, marker="s", s=70, color="#888888",
           zorder=7, edgecolor="white", linewidth=0.8)
ax.scatter(mv_w[0]*100, mv_w[1]*100, marker="s", s=70, color=C1,
           zorder=7, edgecolor="white", linewidth=0.8)
# arrow showing the leftward shift of min-var
ax.annotate("", xy=(mv_w[0]*100, mv_w[1]*100), xytext=(mv_n[0]*100, mv_n[1]*100),
            arrowprops=dict(arrowstyle="->", color=C4, lw=1.6))
ax.text((mv_w[0]+mv_n[0])/2*100, mv_w[1]*100-1.4,
        f"min-variance\nsd {mv_n[0]*100:.1f}% $\\to$ {mv_w[0]*100:.1f}%",
        color=C4, fontsize=8, ha="center", va="top")

# risk-free marker
ax.scatter(0, rf*100, marker="*", s=90, color="#333333", zorder=7)
ax.annotate("3-mo T-bill (rf)", (0, rf*100), textcoords="offset points", xytext=(8, -2), fontsize=8)

ax.set_xlabel("Annualized standard deviation (%)")
ax.set_ylabel("Annualized return (%)")
ax.set_title("Incremental effect of a fine wine sleeve on the efficient frontier",
             loc="left", fontsize=10.5, color=INK, pad=8)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.set_xlim(-0.5, 23)
fig.tight_layout()
fig.savefig("figD_frontier7.png", bbox_inches="tight")
# flatten alpha
from PIL import Image
im=Image.open("figD_frontier7.png").convert("RGBA")
bg=Image.new("RGBA",im.size,(255,255,255,255))
Image.alpha_composite(bg,im).convert("RGB").save("figD_frontier7.png", dpi=(300,300))
print("saved figD_frontier7.png", im.size)
