from __future__ import annotations

import collections
import functools
import math
from typing import (Any, Callable, Iterable, SupportsFloat, SupportsIndex,
                    Type, TypeVar, Union, overload)

import sys
sys.path.append("..")

import cv2
import numpy as np
from utils.logging_cfg import logging
from matplotlib import pyplot as plt

from .functional import cmp_to_key_conversion, paired_map
from .types import Bool, Float, NDArray, Shape, UInt8

Polyline2D = list["Point2D"]
PolygonEdges2D = tuple[Polyline2D, Polyline2D, Polyline2D, Polyline2D]
Bound2D = tuple[Float, Float, Float, Float]
Homography2D = NDArray[Shape["3, 3"], Float]

ImageShape2D = Shape["* width, * height"]
ImageShape3D = Shape["* width, * height, [b, g, r] channels"]
ImageShape = Union[ImageShape2D, ImageShape3D]
GrayImage = NDArray[ImageShape2D, UInt8]
ColorImage = NDArray[ImageShape3D, UInt8]
Image = Union[GrayImage, ColorImage]
ImageMask = NDArray[ImageShape2D, Bool]


class Point2D(np.ndarray):
    _ndtype_: Type = NDArray[Shape["2"], Float]

    def __new__(
        cls, a1: SupportsFloat | Iterable[SupportsFloat], a2: SupportsFloat = None
    ):
        if isinstance(a1, SupportsFloat) and isinstance(a2, SupportsFloat):
            return cls.from_coords(x=a1, y=a2)
        if isinstance(a1, Iterable):
            if a2 is not None:
                logging.warning(f"Ignoring positional argument a2 of type {type(a2)}")
            return cls.from_iter(iter=a1)

    @classmethod
    def from_coords(cls, x: SupportsFloat, y: SupportsFloat):
        return np.array([x, y], dtype=Float).view(cls)

    @classmethod
    def from_iter(cls, iter: Iterable[SupportsFloat]):
        pts: list[SupportsFloat] = list(iter)
        if len(pts) != 2:
            raise ValueError(
                f"Cannot convert sequence of {len(pts)} items to {cls.__class__}"
            )
        # if not all(map(lambda pt: isinstance(pt, np.number), pts)):
        if not all(map(lambda pt: isinstance(pt, SupportsFloat), pts)):
            raise ValueError(
                f"Cannot convert types {type(pts[0])} and {type(pts[1])} to {Float}"
            )

        return cls.__new__(Point2D, *pts)

    @property
    def x(self) -> Float:
        return self[0]

    @property
    def y(self) -> Float:
        return self[1]

    def dist_between(self, other: Point2D) -> Float:
        return Point2D.dist(self, other)

    def __str__(self) -> str:
        return f"(x={self.x}, y={self.y})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: Point2D) -> bool:
        if isinstance(other, self.__class__):
            return self.x == other.x and self.y == other.y
        return super().__eq__(other)

    @classmethod
    def dist(cls, a: Point2D, b: Point2D) -> Float:
        return np.linalg.norm(a - b).astype(Float)

    @classmethod
    def clockwise_sort_cmp_factory(
        cls, center: Point2D
    ) -> Callable[[Point2D, Point2D], bool]:
        @cmp_to_key_conversion  # Python requires comparison functions to return 1, 0 or -1 instead of True or False
        def cmp_fn(a: Point2D, b: Point2D) -> bool:
            """Comparison function for sorting 2D points clockwise, starting from the 12 o'clock.
            Points on the same "hour" will be ordered starting from the ones that are further from the center.
            Adapted from: https://stackoverflow.com/questions/6989100/sort-points-in-clockwise-order

            Parameters
            ----------
            a : Point2D
            b : Point2D

            Returns
            -------
            bool
                Considering a clockwise order, whether `a` appears before `b`.
            """
            # adapted from
            if (a.x - center.x >= 0) and (b.x - center.x < 0):
                return True
            if (a.x - center.x < 0) and (b.x - center.x >= 0):
                return False
            if (a.x - center.x == 0) and (b.x - center.x == 0):
                return (
                    a.y > b.y
                    if (a.y - center.y >= 0) or (b.y - center.y >= 0)
                    else b.y > a.y
                )
            # compute the cross product of vectors (center -> a) x (center -> b)
            det: Float = (a.x - center.x) * (b.y - center.y) - (b.x - center.x) * (
                a.y - center.y
            )
            if det < 0:
                return True
            if det > 0:
                return False

            # points a and b are on the same line from the center
            # check which point is closer to the center
            d1: Float = (a.x - center.x) * (a.x - center.x) + (a.y - center.y) * (
                a.y - center.y
            )
            d2: Float = (b.x - center.x) * (b.x - center.x) + (b.y - center.y) * (
                b.y - center.y
            )
            return d1 > d2

        return cmp_fn


class Rectangle2D:
    def __init__(self, top_left: Point2D, width: Float, height: Float) -> None:
        assert isinstance(top_left, Point2D)

        assert width > 0
        assert height > 0

        self.top_left: Point2D = top_left
        self.width: Float = width
        self.height: Float = height

        self.center: Point2D = Point2D(top_left.x + width / 2, top_left.y + height / 2)

    def __repr__(self):
        return f"tl={self.top_left}, c={self.center}, w={self.width}, h={self.height}"

    def to_polygon(self):
        return Polygon2D(
            [
                self.top_left,
                self.top_left + Point2D(self.width, 0),
                self.top_left + Point2D(self.width, self.height),
                self.top_left + Point2D(0, self.height),
            ]
        )


class Polygon2D(collections.UserList[Point2D]):
    def __init__(self, data: list[Point2D | Iterable[SupportsFloat]]) -> None:
        self.data: list[Point2D] = [
            pt.view(Point2D) if isinstance(pt, Point2D) else Point2D.from_iter(pt)
            for pt in data[0]
        ]

    @classmethod
    def from_str(cls, text: str, sep_d0: str = ",", sep_d1: str = ";"):
        return Polygon2D(
            map(lambda coord: map(float, coord.split(sep_d0)), text.split(sep_d1))
        )

    def to_str(self, sep_d0: str = ",", sep_d1: str = ";") -> str:
        return sep_d1.join(
            map(lambda pt: sep_d0.join(map(str, [pt[0], pt[1]])), self.data)
        )

    @overload
    def __getitem__(self, key: SupportsIndex) -> Point2D:
        ...

    @overload
    def __getitem__(self, key: slice) -> list[Point2D]:
        ...

    def __getitem__(self, key: SupportsIndex | slice) -> Point2D | list[Point2D]:
        if isinstance(key, slice):
            if key.start is None or key.stop is None or key.start <= key.stop:
                return [self.data[i] for i in range(*key.indices(len(self.data)))]
            return [
                self.data[i]
                for i in range(
                    *(slice(key.start, None, key.step)).indices(len(self.data))
                )
            ] + [
                self.data[i]
                for i in range(
                    *(slice(None, key.stop, key.step)).indices(len(self.data))
                )
            ]
        return self.data[key.__index__()]

    def bound_coords(self) -> Bound2D:
        data: list[Point2D] = self.data

        most_left: Float = data[0].x
        most_right: Float = data[0].x
        most_top: Float = data[0].y
        most_bottom: Float = data[0].y

        for pt in data[1:]:
            if pt.x < most_left:
                most_left = pt.x
            if pt.x > most_right:
                most_right = pt.x
            if pt.y > most_bottom:
                most_bottom = pt.y
            if pt.y < most_top:
                most_top = pt.y

        return most_left, most_right, most_bottom, most_top

    def bound_size_from_coords(self, coords: Bound2D) -> tuple[Float, Float]:
        ml, mr, mb, mt = coords
        return mr - ml, mb - mt

    def bound_size(self) -> tuple[Float, Float]:
        return self.bound_size_from_coords(self.bound_coords())

    def bounding_box(self) -> Rectangle2D:
        ml, mr, mb, mt = self.bound_coords()
        return Rectangle2D(Point2D(ml, mt), mr - ml, mb - mt)

    def edges_extremes_idxs(self) -> tuple[int, int, int, int]:
        # width, height = self.bound_size()
        bb = self.bounding_box().to_polygon()
        tl_idx = min(range(len(self)), key=lambda idx: Point2D.dist(bb[0], self[idx]))
        tr_idx = min(range(len(self)), key=lambda idx: Point2D.dist(bb[1], self[idx]))
        bl_idx = min(range(len(self)), key=lambda idx: Point2D.dist(bb[2], self[idx]))
        br_idx = min(range(len(self)), key=lambda idx: Point2D.dist(bb[3], self[idx]))
        return tl_idx, tr_idx, br_idx, bl_idx

    def points_per_edge(self, sort=True) -> PolygonEdges2D:
        extremes_idxs: tuple[int, int, int, int] = self.edges_extremes_idxs()
        if sort:
            extremes_idxs = tuple(sorted(extremes_idxs))

        return (
            self[extremes_idxs[0] : extremes_idxs[1] + 1],
            self[extremes_idxs[1] : extremes_idxs[2] + 1],
            self[extremes_idxs[2] : extremes_idxs[3] + 1],
            self[extremes_idxs[3] : extremes_idxs[0] + 1],
        )

    @classmethod
    def join_edges(cls, edges: PolygonEdges2D) -> list[Point2D]:
        return edges[0][:-1] + edges[1][:-1] + edges[2][:-1] + edges[3][:-1]

    @classmethod
    def shape_from_edges(cls, edges: PolygonEdges2D) -> tuple[int, int]:
        top_edge_dist: Float = sum(paired_map(Point2D.dist, edges[0]))
        right_edge_dist: Float = sum(paired_map(Point2D.dist, edges[1]))
        bottom_edge_dist: Float = sum(paired_map(Point2D.dist, edges[2]))
        left_edge_dist: Float = sum(paired_map(Point2D.dist, edges[3]))

        rect_width, rect_height = tuple(
            map(
                math.ceil,
                [
                    max(top_edge_dist, bottom_edge_dist),
                    max(left_edge_dist, right_edge_dist),
                ],
            )
        )
        return rect_width, rect_height

    def rectifying_homography(
        self, return_rect_shape=False
    ) -> Homography2D | tuple[Homography2D, tuple[int, int]]:
        # bb = self.bounding_box()
        annot_edges: PolygonEdges2D = self.points_per_edge()
        top_edge, right_edge, bottom_edge, left_edge = annot_edges

        rect_shape = self.shape_from_edges(annot_edges)
        rect_width, rect_height = rect_shape

        cum_dist: Float = 0
        top_edge_h: list[Point2D] = [Point2D(cum_dist, 0)]
        for line_dist in paired_map(Point2D.dist, top_edge):
            cum_dist += line_dist
            top_edge_h.append(Point2D(cum_dist, 0))

        cum_dist: Float = 0
        left_edge_h: list[Point2D] = [Point2D(0, rect_height - cum_dist)]
        for line_dist in paired_map(Point2D.dist, left_edge):
            cum_dist += line_dist
            left_edge_h.append(Point2D(0, rect_height - cum_dist))

        cum_dist: Float = 0
        bottom_edge_h: list[Point2D] = [Point2D(rect_width - cum_dist, rect_height)]
        for line_dist in paired_map(Point2D.dist, bottom_edge):
            cum_dist += line_dist
            bottom_edge_h.append(Point2D(rect_width - cum_dist, rect_height))

        cum_dist: Float = 0
        right_edge_h: list[Point2D] = [Point2D(rect_width, cum_dist)]
        for line_dist in paired_map(Point2D.dist, right_edge):
            cum_dist += line_dist
            right_edge_h.append(Point2D(rect_width, cum_dist))

        annot_pts: list[Point2D] = self.join_edges(annot_edges)
        rect_pts: list[Point2D] = self.join_edges(
            (top_edge_h, right_edge_h, bottom_edge_h, left_edge_h)
        )

        homo_ret: tuple[Homography2D, Any] = cv2.findHomography(
            np.array(annot_pts, dtype=int), np.array(rect_pts, dtype=int)
        )
        H, _ = homo_ret

        if return_rect_shape:
            return H, rect_shape
        return H

    def sort_clockwise(self, data=None, **kw):
        """Sorts polygon points

        Inplace operation
        """
        bb: Rectangle2D = self.bounding_box()
        cmp_fn = Point2D.clockwise_sort_cmp_factory(bb.center)
        key_fn = functools.cmp_to_key(cmp_fn)
        if data is None:
            self.sort(key=key_fn, **kw)
            return self
        return sorted(data, key=key_fn, **kw)

    def get_closest_point_idx(self, reference_point: Point2D) -> int:
        return min(
            range(len(self.data)),
            key=lambda idx: Point2D(reference_point).dist_between(self.data[idx]),
        )

    def set_starting_point(self, starting_point: Point2D):
        starting_idx = self.get_closest_point_idx(starting_point)
        return self.set_starting_point_from_idx(starting_idx)

    def set_starting_point_from_idx(self, idx: int):
        self.data = self.data[idx:] + self.data[:idx]
        return self

    def clip(self, x_max: Float, y_max: Float, x_min=0, y_min=0):
        self.data = [
            Point2D(np.clip(x, x_min, x_max), np.clip(y, y_min, y_max))
            for x, y in self.data
        ]
        return self

    def mask(self, shape):
        rewarp_roi_mask = np.zeros(shape[:2])
        # breakpoint()
        rewarp_roi_mask = cv2.fillPoly(
            rewarp_roi_mask, np.array([self.data], dtype=int), 255
        )
        return rewarp_roi_mask.astype(np.uint8)

    def expand_from_center(self, scale: float):
        raise NotImplementedError()


def mask_overlay(img_bg: Image, img_fg: Image, mask: ImageMask) -> Image:
    if img_bg.shape != img_fg.shape:
        raise ValueError(
            f"Input images shapes do not match: {img_bg.shape} vs {img_fg.shape}"
        )
    shape_2d = img_bg.shape[:2]

    if len(mask.shape) != 2 and (len(mask.shape) != 3 or mask.shape[-1] != 1):
        raise ValueError(
            "\n".join((
                f"Invalid shape for mask {mask.shape}.",
                "Expected one of:" f"{shape_2d} or {(shape_2d[0], shape_2d[1], 1)}",
            ))
        )
    mask = mask.reshape(shape_2d)

    mask = np.where(mask == 255, 255, 0)
    mask_3d = cv2.merge((mask, mask, mask))

    # breakpoint()
    # mask_inv = cv2.bitwise_not(mask)
    # mask_inv_3d = cv2.merge((mask_inv, mask_inv, mask_inv))
    # return cv2.add(
    #     cv2.bitwise_and(img_bg,img_bg,mask=mask_inv_3d),
    #     cv2.bitwise_and(img_fg,img_fg,mask=mask_3d)
    # )

    # breakpoint()
    # return cv2.add(
    #     cv2.bitwise_and(img_fg,mask_3d),
    #     cv2.bitwise_and(img_bg,mask_inv_3d)
    # )

    return np.where(mask_3d == 0, img_bg, img_fg)


IM = TypeVar("IM", ColorImage, GrayImage)


class Rectifier:
    canvas_img: IM
    roi_polygon: Polygon2D
    H: Homography2D
    rect_shape: tuple[int, int]
    rewarp_roi_mask: ImageMask

    def __init__(self, canvas_img: IM, roi_polygon: Polygon2D):
        self.roi_polygon = roi_polygon
        self.canvas_img = canvas_img
        self.H, self.rect_shape = roi_polygon.rectifying_homography(
            return_rect_shape=True
        )
        self.rewarp_roi_mask = self.roi_polygon.mask(self.canvas_img.shape)

    def rectify(self, borderValue=(0, 0, 0)) -> IM:
        return cv2.warpPerspective(
            self.canvas_img, self.H, self.rect_shape, borderValue=borderValue
        )

    def rewarp(self, rect_img: IM) -> IM:
        assert rect_img.shape[1::-1] == self.rect_shape
        rewarped_roi_img = cv2.warpPerspective(
            rect_img, self.H, self.canvas_img.shape[1::-1], flags=cv2.WARP_INVERSE_MAP
        )

        reconstructed_img: IM = mask_overlay(
            img_bg=self.canvas_img, img_fg=rewarped_roi_img, mask=self.rewarp_roi_mask
        )
        reforce_mask: GrayImage = cv2.bitwise_and(
            self.rewarp_roi_mask,
            np.where(
                cv2.cvtColor(reconstructed_img, cv2.COLOR_BGR2GRAY) == 0, 255, 0
            ).astype(np.uint8),
        )

        return mask_overlay(
            img_fg=self.canvas_img, img_bg=reconstructed_img, mask=reforce_mask
        )


def debug_rewarp(rectifier, output_prefix):
    warped_roi = rectifier.rectify()

    reconstructed_img = rectifier.rewarp(warped_roi)
    reconstructed_img_gray = rectifier.rewarp(warped_roi * 0 + 120)

    diff_img = np.abs(reconstructed_img - rectifier.canvas_img)

    plot_rets = []
    plot_rets.append(
        f"{cv2.imwrite(output_prefix+'rewarped.jpg', reconstructed_img) = }"
    )
    plot_rets.append(f"{cv2.imwrite(output_prefix+'warped.jpg', warped_roi) = }")

    fig = plt.figure(constrained_layout=True)

    img_dict = dict(
        zip(
            iter("ABCDE"),
            map(
                lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB),
                (
                    rectifier.canvas_img,
                    reconstructed_img,
                    reconstructed_img_gray,
                    diff_img,
                    warped_roi,
                ),
            ),
        )
    )
    titles_dict = dict(
        zip(
            iter("ABCDE"),
            (
                "Input image",
                "Rewarp with original content",
                "Rewarp with constant gray content",
                "Rewarp error",
                "Normalized Input",
            ),
        )
    )
    # mosaic_layout = (
    #     "AB;DE"
    #
    #     "ABD;EEE")
    mosaic_layout = "AB;ED"
    width_ratios = [1, 1]
    height_ratios = [1, 1]
    if not (rectifier.canvas_img.shape[1] > rectifier.canvas_img.shape[0]):
        height_ratios = [rectifier.canvas_img.shape[1], warped_roi.shape[1]]
        fig.set_size_inches(16, 18)

    else:
        fig.set_size_inches(16, 9)

    scale = 1.2 * 1920 / (fig.get_figwidth() * fig.get_dpi())
    fig.set_dpi(scale * fig.get_dpi())
    ax_dict = fig.subplot_mosaic(
        mosaic_layout,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        gridspec_kw={
            "wspace": 0,
            "hspace": 0,
        },
    )

    print(f"{ax_dict = }")
    for k in ax_dict.keys():
        ax = ax_dict[k]
        ax.imshow(img_dict[k])
        ax.set_title(titles_dict[k])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    print(
        fig.dpi,
        fig.get_size_inches(),
        tuple(map(lambda c: c * fig.dpi, fig.get_size_inches())),
    )

    # resolution_value = int(2e3)
    # fig.tight_layout()
    plot_rets.append(
        f"{fig.savefig(output_prefix+'plot.jpg', pad_inches=0, format='png') = }"
    )
    return plot_rets
