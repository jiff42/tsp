import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def combine_four_images(images, captions, output_path='combined.png'):
    """
    将四张图像拼接成2x2排列，并在每张图下方添加注释。
    
    参数：
        images: list[str]，四个图像路径，例如 ['a.png', 'b.png', 'c.png', 'd.png']
        captions: list[str]，四个对应的注释，例如 ['xxx', 'yyy', 'zzz', 'www']
        output_path: str，输出拼接后图像的保存路径
    """
    assert len(images) == 4 and len(captions) == 4, "必须提供4个图像和4个注释"

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for i, (img_path, caption) in enumerate(zip(images, captions)):
        img = mpimg.imread(img_path)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"({chr(97 + i)}) {caption}", fontsize=12, y=-0.15)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"拼接完成，已保存为 {output_path}")

if __name__ == "__main__":
    combine_four_images(
        images=[
            "figs_tsplib_eil76/eil76_opt_route.png",
            "exp7/ga_route.png",
            "exp7/sa_route.png",
            "exp7/aco_route.png",
        ],
        captions=[
            "best route",
            "ga route",
            "sa route",
            "aco route",
        ],
        output_path="figs_combine/combined_eil76.png",
    )