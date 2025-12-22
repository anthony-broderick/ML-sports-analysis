from math import hypot, dist


def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 +x2) / 2), int(y2))

def measure_xy_distance(p1, p2):
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])

def get_center_of_bbox(bbox):
    return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

def get_extreme_court_key_points(court_keypoints):
    """
    # Compute the centroid of the court
    x_coords = [p[0] for p in court_keypoints]
    y_coords = [p[1] for p in court_keypoints]
    court_center = (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

    # Find two points closest to the centroid
    closest_index_one = None
    closest_index_two = None
    min_dist_one = float('inf')
    min_dist_two = float('inf')

    for i, point in enumerate(court_keypoints):
        x, y = point
        dist = hypot(x - court_center[0], y - court_center[1])

        if dist < min_dist_one:
            min_dist_two = min_dist_one
            closest_index_two = closest_index_one
            min_dist_one = dist
            closest_index_one = i
        elif dist < min_dist_two:
            min_dist_two = dist
            closest_index_two = i

    # Collect indices of the two closest points
    extreme_indices = [closest_index_one, closest_index_two]

    # Filter points to the left of the centroid
    leftmost_indices = [i for i, p in enumerate(court_keypoints) if p[0] < court_center[0]]

    # Find two leftmost points furthest from the centroid
    furthest_left_index_one = None
    furthest_left_index_two = None
    max_dist_one = float('-inf')
    max_dist_two = float('-inf')

    for i in leftmost_indices:
        x, y = court_keypoints[i]
        dist = hypot(x - court_center[0], y - court_center[1])

        if dist > max_dist_one:
            max_dist_two = max_dist_one
            furthest_left_index_two = furthest_left_index_one
            max_dist_one = dist
            furthest_left_index_one = i
        elif dist > max_dist_two:
            max_dist_two = dist
            furthest_left_index_two = i

    # Add the two leftmost furthest points (if they exist)
    if furthest_left_index_one is not None:
        extreme_indices.append(furthest_left_index_one)
    if furthest_left_index_two is not None:
        extreme_indices.append(furthest_left_index_two)
    """
    extreme_indices = [10,7,2,0]

    return extreme_indices

def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    closest_distance = float('inf')
    key_point_ind = keypoint_indices[0]
    
    for keypoint_index in keypoint_indices:
        x, y = keypoints[keypoint_index]  # unpack tuple directly
        
        distance = distance = abs(point[1] - y)  # vertical distance only
        # distance = dist(point, (x, y))  # Euclidean distance
        
        if distance < closest_distance:
            closest_distance = distance
            key_point_ind = keypoint_index

    return key_point_ind



def get_height_of_bbox(bbox):
    return bbox[3] - bbox[1]