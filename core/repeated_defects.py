"""Find similar compact defects that recur at a fixed scan position."""

from collections import defaultdict
from itertools import product
from math import floor, hypot, log


def repeated_defects(photos, *, tolerance=0.0028, minimum=3):
    """Return groups of (file, candidate) for photos with comparable geometry."""
    groups = []
    grid = defaultdict(list)
    neighbors = tuple(product((-1, 0, 1), repeat=2))
    for path, (width, height), items in photos:
        for item in items:
            if item.dust_type not in ("bright", "dark", "infrared") or item.aspect_ratio > 3:
                continue
            x, y = item.center_x / width, item.center_y / height
            size = (item.width / width, item.height / height)
            aspect = width / height
            bx, by = floor(x / tolerance), floor(y / tolerance)
            matches = []
            for dx, dy in neighbors:
                for index in grid[item.dust_type, bx + dx, by + dy]:
                    anchor, anchor_size, anchor_aspect, members = groups[index]
                    distance = hypot(x - anchor[0], y - anchor[1])
                    if (path not in members and distance <= tolerance
                            and abs(log(aspect / anchor_aspect)) <= 0.035
                            and all(0.45 <= a / b <= 2.2 for a, b in zip(size, anchor_size))):
                        matches.append((distance, index))
            if matches:
                index = min(matches)[1]
                groups[index][3][path] = item
            else:
                grid[item.dust_type, bx, by].append(len(groups))
                groups.append(((x, y), size, aspect, {path: item}))
    return sorted((list(members.items()) for _, _, _, members in groups if len(members) >= minimum),
                  key=len, reverse=True)
