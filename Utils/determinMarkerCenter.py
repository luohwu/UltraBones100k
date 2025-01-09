import numpy as np

def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0]) ** 2 + (cy - p1[1]) ** 2)
    return ((cx, cy), radius)


center, radius = define_circle((-30.9269 ,-162.692), (-30.244,-161.096 ), (-28.2643 ,-163.449 ))
print(f"center {center}, radius: {radius}")

center, radius = define_circle((22.6658 ,-142.324 ), (24.6977 ,-144.361  ), (21.7506  ,-144.044  ))
print(f"center {center}, radius: {radius}")

center, radius = define_circle((-17.8104  ,-117.367 ), (-16.0749 ,-117.406  ), (-16.0246 ,-119.898 ))
print(f"center {center}, radius: {radius}")

center, radius = define_circle((19.8019  ,-111.205 ), (22.0445 ,-112.279  ), (21.5708  ,-113.712  ))
print(f"center {center}, radius: {radius}")