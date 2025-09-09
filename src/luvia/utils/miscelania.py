    
class MEH:
    def __init__(self):
        pass
    def subsubplot9(self, ax_spec, fig, data, cmap):
        gs_nested = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=ax_spec, wspace=0.05, hspace=0.05)
        axes = []

        for i in range(3):
            for j in range(3):
                idx = i * 3 + j
                ax = fig.add_subplot(gs_nested[i, j], facecolor="black")
                ax.set_facecolor('black')
                ax.imshow(data[idx], cmap=cmap)
                ax.set_title("{}_{}".format(self.name, "subplots"), color='white')
                ax.axis('off')
                axes.append(ax)

        return axes

    def createfinal_char_image(self, char_num, line_num):
        fig = plt.figure(figsize=(10, 12), facecolor="black")
        gs = gridspec.GridSpec(4, 3, height_ratios=[1.75, 1.75, 1, 1], hspace=0.1, wspace=0.05)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])
        ax7 = fig.add_subplot(gs[2, :])
        ax8 = fig.add_subplot(gs[3, :])

        all_axes = [ax1, ax2, ax3, ax4]
        for ax in all_axes:
            ax.set_facecolor('white')
            ax.tick_params(colors='white')
            ax.xaxis.set_tick_params(labelcolor='white', labelsize=6)
            ax.yaxis.set_tick_params(labelcolor='white', labelsize=6)
            ax.axis('on')
        ax5.set_facecolor("black")
        ax6.set_facecolor("black")
        char_data = self.image_objects["lines"][f"line-{line_num}"]["characters"][f"character-{char_num}"]

        ax1.imshow(char_data["original"], cmap="gray")
        ax1.set_title("{}_{}".format(self.name, "original"), color='white')
        ax2.imshow(char_data["saliency"], cmap='hot')
        ax2.set_title("{}_{}".format(self.name, "saliency"), color='white')
        ax3.imshow(char_data["gb_grad"], cmap='inferno')
        ax3.set_title("{}_{}".format(self.name, "gradient_backpropagation"), color='white')
        ax6.imshow(char_data["sensitivity"], cmap='coolwarm')
        ax6.set_title("{}_{}".format(self.name, "sensitivity"), color='white')

        self.subsubplot9(gs[1, 0], fig, char_data["conv1"], cmap="gray")
        self.subsubplot9(gs[1, 1], fig, char_data["act1"], cmap="magma")
        plt.tight_layout()
        plt.savefig(f"{self.output_folder}/meh.png", bbox_inches='tight')
        plt.close()