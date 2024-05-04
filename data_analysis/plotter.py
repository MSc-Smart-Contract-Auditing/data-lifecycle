from matplotlib import pyplot as plt
from common.directories import FIGURES_DIR

plt.rcParams["font.size"] = 16  # Default font size for all text
plt.rcParams["axes.labelsize"] = 16  # Font size for x and y labels
plt.rcParams["xtick.labelsize"] = 14  # Font size for x-tick labels
plt.rcParams["ytick.labelsize"] = 14  # Font size for y-tick labels
plt.rcParams["legend.fontsize"] = 14  # Font size for legend
plt.rcParams["axes.titlesize"] = 18  # Font size for the title


class Plotter:
    def __init__(self, dataset_name):
        self.dataset_dir = FIGURES_DIR / dataset_name
        self.dataset_dir.mkdir(exist_ok=True)

    def save_plot(self, name):
        plt.savefig(
            self.dataset_dir / f"{name}.pdf", bbox_inches="tight", dpi=800, format="pdf"
        )
        plt.show()
        plt.close()
