import pickle as pkl
import os
import matplotlib.pyplot as plt

def format_metrics(metrics, split):
    """Format metrics for logging."""
    result = f"{split} MR: {metrics['MR'].item():.3f} | "
    result += f"MRR: {metrics['MRR'].item():.3f} | "
    result += f"H10: {metrics['HITS10']:.3f} | "
    result += f"H3: {metrics['HITS3']:.3f} | "
    result += f"H1: {metrics['HITS1']:.3f} |"
    result += f"MRR_XX: {metrics['MRR_XX'].item():.3f}"
    return result

def plotter(x, name, model_name, _save, _show, save_dir):
    y = [i+1 for i in range(len(x))]
    fig, ax = plt.subplots()
    ax.plot(y, x)
    ax.scatter(y, x, s=40, color = 'red')
    ax.set_xlabel('Number of Samples', fontsize = 13)
    ax.set_ylabel(name, fontsize = 13)
    ax.set_title(name+' metrics', fontsize = 20)
    ax.grid()

    plt.savefig(os.path.join(save_dir, model_name) + '_' + name + '.png', bbox_inches='tight') if _save else None
    plt.show() if _show else None
