def generate_anchors(feature_size: int, image_size: float, scales: list[float], aspect_ratios: list[float]) -> list[list[float]]:
    """
    Returns a list of [x1, y1, x2, y2] anchor boxes.
    """
    result = []
    stride = image_size / feature_size
    for i in range(feature_size):
        for j in range(feature_size):
            for s in scales:
                for r in aspect_ratios:
                    cx = (j + 0.5) * stride
                    cy = (i + 0.5) * stride

                    w = s * r**(1/2)
                    h = s / r**(1/2)

                    anchor_box = [cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0]
                    result.append(anchor_box)
    return result
                    
    pass