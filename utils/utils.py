import matplotlib.pyplot as plt


def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()


import cv2
import numpy as np

def compute_optical_flow(frame1, frame2, visualize=False):
    """
    Compute dense optical flow between two frames using Farnebäck method.

    Args:
        frame1 (np.ndarray): First frame (grayscale or RGB).
        frame2 (np.ndarray): Second frame (grayscale or RGB).
        visualize (bool): If True, display the magnitude of flow.

    Returns:
        flow (np.ndarray): Optical flow with shape (H, W, 2) containing (flow_x, flow_y).
        magnitude (np.ndarray): Optical flow magnitude.
    """
    # Ensure frames are grayscale
    if len(frame1.shape) == 3:
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    if len(frame2.shape) == 3:
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        frame1, frame2, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    # Compute flow magnitude
    flow_x, flow_y = flow[..., 0], flow[..., 1]
    magnitude = np.sqrt(flow_x**2 + flow_y**2)

    if visualize:
        import matplotlib.pyplot as plt
        plt.imshow(magnitude, cmap='hot')
        plt.title('Optical Flow Magnitude')
        plt.axis('off')
        plt.show()

    return flow, magnitude