# # -*- coding: utf-8 -*-
# # @Author  : Heisenberg
# # @Time    : 2023/5/4 20:45
# # @Software: PyCharm
#
# import geographiclib.geodesic as geo
# import torch
#
# a = 6378137  # equatorial radius
# f = 1 / 298.257223563  # flattening
#
#
# def distance_calculate(coordinates: torch.Tensor, target_coordinates: torch.Tensor) -> torch.Tensor:
#     """
#     Calculate the distance between two tensor coordinates.
#     :param coordinates: tensor positioning coordinates.
#     :param target_coordinates: tensor real coordinates.
#     :return: distance between two tensor coordinates.
#     """
#     # Input coordinates
#     coordinates = coordinates.squeeze(2)
#     target_coordinates = target_coordinates.squeeze(2)
#     lat1, lon1, h1 = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
#     lat2, lon2, h2 = target_coordinates[:, 0], target_coordinates[:, 1], target_coordinates[:, 2]
#
#     # Calculate distance using Vincenty algorithm
#     length = coordinates.shape[0]
#     g = geo.Geodesic(a, f)
#     s12 = torch.zeros(length)
#     a12 = torch.zeros(length)
#     for i in range(length):
#         # s12: the distance from the first point to the second in meters
#         s12[i] = torch.tensor(g.Inverse(lat1[i], lon1[i], lat2[i], lon2[i])['s12'])
#         # a12: the azimuth from the first point to the second in degrees
#         a12[i] = torch.tensor(g.Inverse(lat1[i], lon1[i], lat2[i], lon2[i])['a12'])
#     distance = torch.sqrt((s12 * torch.cos(torch.deg2rad(a12))) ** 2 + (h1 - h2) ** 2)
#     return distance
