# Plot shear/bending/tangent/deflection of a beam
# Print extreme values

import numpy as np
import matplotlib.pyplot as plt

from PP import PiecewisePolynomial


BEAM_LENGTH = 1250.  # length of the beam


def solve_beam(x1, x2, p1, p2, point_loads):
    """
    Solves beam problem and returns four tuples (max_argx, max_val):
    - Reaction force (N) (max_x==0)
    - Shear force (N)
    - Bending moment (N⋅mm)
    - Deflection (mm)
    """
    #Make sure the values are alligned
    assert x2 > x1 and x1 <= p1 < p2 <= x2
    # Solve for reaction forces
    total_vert = sum(f for x, f in point_loads if x1 <= x <= x2)
    moment_about_p1 = sum(f * (x - p1) for x, f in point_loads if x1 <= x <= x2)
    f2 = -moment_about_p1 / (p2 - p1)
    f1 = -total_vert - f2
    print(f"Reaction force at support at 25m: {f1} Reaction force at support at 1225m: {f2}")
    
    # Key points
    keypoints_dict = {p1: [f1, 0], p2: [f2, 0]}
    for x, f in point_loads:
        if x1 <= x <= x2:
            keypoints_dict.setdefault(x, [0, 0])[0] += f
    keypoints = [(x1, 0, 0)] + sorted((x, *vals) for x, vals in keypoints_dict.items()) + [(x2, 0, 0)]
    
    # SFD, BMD, slope, and deflection calculations
    poly_keypoints, poly_pieces = [x1], []
    cul, cld = 0.0, 0.0
    for x, pl, dul in keypoints[1:]:
        b, m = cld - cul * poly_keypoints[-1], cul
        poly_keypoints.append(x)
        poly_pieces.append([b, m])
        cul += dul
        cld += cul * (x - poly_keypoints[-2]) + pl
    sfd = PiecewisePolynomial(poly_keypoints, poly_pieces)
    bmd = sfd.integrate_segments()
    xs = np.linspace(x1 + 1e-12, x2 - 1e-12, int(x2 - x1) + 1)
    return (
        (xs, max(f1, f2)),
        (xs, np.abs(sfd.compute_all(xs))),
        (xs, bmd.compute_all(xs))
    )
    





def get_beam_responses(train_position, train_load):
    # Define joint positions on the train
    joint_positions = [52, 228, 392, 568, 732, 908]
    # Call the solve_beam function with joint positions and load
    return solve_beam(
        0, BEAM_LENGTH,
        0.5 * BEAM_LENGTH - 600, 0.5 * BEAM_LENGTH + 600,
        [[joint_positions[i] + train_position, -1 * train_load[i]] for i in range(len(joint_positions))]
    )

def plot_beam_envelope(load, plot=True):
    # Define start and end positions for the train
    start_x, end_x = -960, int(BEAM_LENGTH)

    # Create an array of positions for the train
    positions = np.linspace(start_x, end_x, end_x - start_x + 1, dtype=np.float64)

    # Initialize arrays for maximum values
    max_shear, max_bending = np.array([]), np.array([])
    max_shear_pos, max_bending_pos = start_x, start_x

    reactions_xs = []
    reactions_vals = []
    # Loop over each train position to find max values
    for pos in positions:
        print(pos)
        (reactions_x, reactions_val), (shear_x, shear_val), (bending_x, bending_val) = get_beam_responses(pos, load)
        reactions_xs.append(pos)
        reactions_vals.append(reactions_val)
        max_shear = np.maximum(max_shear, shear_val) if max_shear.size else shear_val
        max_bending = np.maximum(max_bending, bending_val) if max_bending.size else bending_val

        # Update positions if new max values are found
        if plot:
            if np.amax(max_shear) == np.amax(shear_val):
                max_shear_pos = pos
            if np.amax(max_bending) == np.amax(bending_val):
                max_bending_pos = pos

    # Plot the results if requested
    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 4))
        ax1.set_title("Max Shear (N)")
        ax1.plot(shear_x, max_shear, '-')
        ax2.set_title("Max Bending (×10³ N⋅mm)")
        ax2.plot(bending_x, 0.001 * max_bending, '-')
        ax3.plot(reactions_xs, reactions_vals)
        ax3.set_title("Max reaction forces for every train location (N)")
        plt.tight_layout()
        plt.show()

    # Print the maximum values
    print("Max reaction", max(reactions_vals))
    print("Max shear", np.amax(max_shear), "N", "at train_x =", max_shear_pos, "at x =", shear_x[np.argmax(max_shear)])
    print("Max bmx", np.amax(max_bending), "N⋅mm", "at train_x =", max_bending_pos, "at x =", bending_x[np.argmax(max_bending)])

def only_max(length, load):
    # Define positions for maximum shear and bending
    max_shear_x, max_bending_x = -26.0, 207.0 if length == 1250 else (0, 0)

    # Get responses for maximum shear and bending
    (_, _), (_, max_shear), (_) = get_beam_responses(max_shear_x, load)
    (_, _), (_, _), (_, max_bending) = get_beam_responses(max_bending_x, load)

    # Print the maximum shear and bending values
    print(np.amax(max_shear))
    print(np.amax(max_bending))

    # Return the max shear and bending values
    return max_shear, max_bending

if __name__ == "__main__":
    #Reaction forces printed out/Diagrams/ and tha max forces calculated
    # Define the load on the train
    # Execute the only_max function with specified beam length and load
    front_car = 400.0
    load_case = [90, 90, 90/1.35, 90/1.35, 90/1.35, 90/1.35]
    load_case = [front_car/6, front_car/6, front_car/6, front_car/6, front_car/6, front_car/6]
    plot_beam_envelope(load_case)