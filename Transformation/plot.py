# import plotly.graph_objects as go
# 
# Define line_world_coords_3D and apply the transformation.
# raw_lines = [
#     [[0., 54.16, 0.], [16.5, 54.16, 0.]],
#     [[16.5, 13.84, 0.], [16.5, 54.16, 0.]],
#     [[16.5, 13.84, 0.], [0., 13.84, 0.]],
#     [[88.5, 54.16, 0.], [105., 54.16, 0.]],
#     [[88.5, 13.84, 0.], [88.5, 54.16, 0.]],
#     [[88.5, 13.84, 0.], [105., 13.84, 0.]],
#     [[0., 37.66, 2.44], [0., 30.34, 2.44]],
#     [[0., 37.66, 0.], [0., 37.66, 2.44]],
#     [[0., 30.34, 0.], [0., 30.34, 2.44]],
#     [[105., 37.66, 2.44], [105., 30.34, 2.44]],
#     [[105., 30.34, 0.], [105., 30.34, 2.44]],
#     [[105., 37.66, 0.], [105., 37.66, 2.44]],
#     [[52.5, 0., 0.], [52.5, 68, 0.]],
#     [[0., 68., 0.], [105., 68., 0.]],
#     [[0., 0., 0.], [0., 68., 0.]],
#     [[105., 0., 0.], [105., 68., 0.]],
#     [[0., 0., 0.], [105., 0., 0.]],
#     [[0., 43.16, 0.], [5.5, 43.16, 0.]],
#     [[5.5, 43.16, 0.], [5.5, 24.84, 0.]],
#     [[5.5, 24.84, 0.], [0., 24.84, 0.]],
#     [[99.5, 43.16, 0.], [105., 43.16, 0.]],
#     [[99.5, 43.16, 0.], [99.5, 24.84, 0.]],
#     [[99.5, 24.84, 0.], [105., 24.84, 0.]]
# ]
# Apply the transformation: subtract 52.5 from x and 34 from y.
# line_world_coords_3D = [
#     [[x1 - 52.5, y1 - 34, z1], [x2 - 52.5, y2 - 34, z2]]
#     for [[x1, y1, z1], [x2, y2, z2]] in raw_lines
# ]
# 
# Create the 3D figure.
# fig = go.Figure()
# 
# Plot each line segment
# for seg in line_world_coords_3D:
#     x_line = [seg[0][0], seg[1][0]]
#     y_line = [seg[0][1], seg[1][1]]
#     z_line = [seg[0][2], seg[1][2]]
#     fig.add_trace(go.Scatter3d(
#         x=x_line, y=y_line, z=z_line,
#         mode='lines',
#         line=dict(width=4),
#         name='Line Segment'
#     ))
# 
# 1. Gather all x, y, z values
# x_vals = []
# y_vals = []
# z_vals = []
# for seg in line_world_coords_3D:
#     x_vals.extend([seg[0][0], seg[1][0]])
#     y_vals.extend([seg[0][1], seg[1][1]])
#     z_vals.extend([seg[0][2], seg[1][2]])
# 
# 2. Find the global min and max
# global_min = min(x_vals + y_vals + z_vals)
# global_max = max(x_vals + y_vals + z_vals)
# 
# 3. Update layout with equal numeric ranges and a cubic aspect ratio
# fig.update_layout(
#     scene=dict(
#         xaxis=dict(range=[global_min, global_max]),
#         yaxis=dict(range=[global_min, global_max]),
#         zaxis=dict(range=[global_min, global_max]),
#         aspectmode='cube'   # Ensures a 1:1:1 aspect ratio
#     ),
#     title='3D Plot with Equal Axis Ranges'
# )
# 
# fig.show()
# 
# 
# 
# 



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define line coordinates
lines_coords = [[[0., 54.16, 0.], [16.5, 54.16, 0.]],
                [[16.5, 13.84, 0.], [16.5, 54.16, 0.]],
                [[16.5, 13.84, 0.], [0., 13.84, 0.]],
                [[88.5, 54.16, 0.], [105., 54.16, 0.]],
                [[88.5, 13.84, 0.], [88.5, 54.16, 0.]],
                [[88.5, 13.84, 0.], [105., 13.84, 0.]],
                [[0., 37.66, -2.44], [0., 30.34, -2.44]],
                [[0., 37.66, 0.], [0., 37.66, -2.44]],
                [[0., 30.34, 0.], [0., 30.34, -2.44]],
                [[105., 37.66, -2.44], [105., 30.34, -2.44]],
                [[105., 30.34, 0.], [105., 30.34, -2.44]],
                [[105., 37.66, 0.], [105., 37.66, -2.44]],
                [[52.5, 0., 0.], [52.5, 68, 0.]],
                [[0., 68., 0.], [105., 68., 0.]],
                [[0., 0., 0.], [0., 68., 0.]],
                [[105., 0., 0.], [105., 68., 0.]],
                [[0., 0., 0.], [105., 0., 0.]],
                [[0., 43.16, 0.], [5.5, 43.16, 0.]],
                [[5.5, 43.16, 0.], [5.5, 24.84, 0.]],
                [[5.5, 24.84, 0.], [0., 24.84, 0.]],
                [[99.5, 43.16, 0.], [105., 43.16, 0.]],
                [[99.5, 43.16, 0.], [99.5, 24.84, 0.]],
                [[99.5, 24.84, 0.], [105., 24.84, 0.]]]


# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each line
for line in lines_coords:
    x, y, z = zip(*line)  # Unpack coordinates
    ax.plot(x, y, z, marker='o', linestyle='-')

# Set labels
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Show plot
plt.show()
