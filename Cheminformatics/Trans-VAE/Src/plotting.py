def plot_training_curves(history, beam_every=None, save_path=None):
    import numpy as np, matplotlib.pyplot as plt

    ep = np.arange(1, len(history["val_loss_tf"]) + 1)
    tr_loss = np.array(history["train_loss"], dtype=float)
    va_loss = np.array(history["val_loss_tf"], dtype=float)
    tr_acc  = np.array(history["train_acc"], dtype=float)
    va_acc  = np.array(history["val_acc_tf"], dtype=float)
    beam_acc = np.array(history["val_acc_beam"], dtype=float)
    val_valid = np.array(history["val_valid"], dtype=float)
    val_lev   = np.array(history["val_lev"], dtype=float)

    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    ax1, ax2, ax3, ax4, ax5, ax6 = axs.flat

    # TF loss
    ax1.plot(ep, va_loss, label="val TF", lw=2)
    ax1.plot(ep, tr_loss, label="train", alpha=0.6)
    ax1.set_title("Loss (TF)"); ax1.set_ylabel("CE+KL"); ax1.legend()

    # TF accuracy
    ax2.plot(ep, va_acc, label="val TF", lw=2)
    ax2.plot(ep, tr_acc, label="train", alpha=0.6)
    ax2.set_title("Token Acc (TF)"); ax2.legend()

    # Beam accuracy
    m_b = ~np.isnan(beam_acc)
    ax3.plot(ep[m_b], beam_acc[m_b], marker="o", lw=2)
    ax3.set_title("Token Acc (Beam)"); ax3.set_ylim(0, 1)

    # Validity (beam)
    m_v = ~np.isnan(val_valid)
    ax4.plot(ep[m_v], val_valid[m_v], marker="o", lw=2)
    ax4.set_title("Validity Ratio (Beam)"); ax4.set_ylim(0, 1)

    # Levenshtein (beam)
    m_l = ~np.isnan(val_lev)
    ax5.plot(ep[m_l], val_lev[m_l], marker="o", lw=2)
    ax5.set_title("Levenshtein (Beam)")

    # mark beam epochs
    if beam_every:
        for ax in axs.flat:
            for e in range(beam_every, len(ep)+1, beam_every):
                ax.axvline(e, color="k", alpha=0.08, lw=1)

    for ax in axs.flat: ax.grid(alpha=0.25)
    axs[1,0].set_xlabel("epoch"); axs[1,1].set_xlabel("epoch"); axs[1,2].set_xlabel("epoch")
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150); print(f"[plot] saved to {save_path}")
    return fig, axs
