from copy import deepcopy
import numpy as np

class PiecewisePolynomial():
    def __init__(self, breakpoints, segment_coeffs):
        # Constructor: Initializes a PiecewisePolynomial object.
        # 'breakpoints' are the points at which the polynomial segments meet.
        # 'segment_coeffs' are the coefficients of the polynomial segments.
        self.segment_count = len(breakpoints) - 1  # Total number of polynomial segments.
        self.breakpoints = deepcopy(sorted(breakpoints))  # Sort and store breakpoints.
        self.segment_coeffs = deepcopy(segment_coeffs)  # Store the polynomial coefficients.

    def __len__(self):
        # Magic method to return the number of polynomial segments.
        return self.segment_count

    @staticmethod
    def _compute_value(coefficients, point):
        # Computes the value of a single polynomial segment at a given point using numpy's polyval.
        # 'coefficients' are reversed because numpy expects them in descending order.
        return np.polyval(coefficients[::-1], point)

    def _locate_segment(self, point):
        # Locates which segment the value point belongs to among the breakpoints.
        # Implements a binary search algorithm for efficiency.
        lower_bound, upper_bound = 0, len(self.breakpoints) - 2  # Setting search bounds.
        while lower_bound <= upper_bound:
            midpoint = (lower_bound + upper_bound) // 2
            if self.breakpoints[midpoint] <= point < self.breakpoints[midpoint + 1]:
                return midpoint
            elif point < self.breakpoints[midpoint]:
                upper_bound = midpoint - 1
            else:
                lower_bound = midpoint + 1
        return -1  # If point is not in any segment, return -1.

    def compute(self, point):
        # Computes the PiecewisePolynomial at a given point.
        segment_idx = self._locate_segment(point)
        if segment_idx is not None:
            return self._compute_value(self.segment_coeffs[segment_idx], point)
        return None

    def compute_all(self, points):
        # Computes the PiecewisePolynomial for a list of points.
        return np.array([self.compute(point) for point in points])
    
    def plot_data(self, num_points=1000):
        # Generates x and y points for plotting the PiecewisePolynomial.
        total_range = self.breakpoints[-1] - self.breakpoints[0]
        x_vals, y_vals = [], []

        for i in range(len(self)):
            # Iterating over each polynomial segment.
            start, end = self.breakpoints[i], self.breakpoints[i+1]
            segment_range = end - start
            num_segment_points = int(np.round(num_points * (segment_range / total_range)))

            # Generate x-values and compute the segment value at these points.
            x_segment_vals = np.linspace(start, end, num_segment_points, endpoint=(i == len(self) - 1))
            x_vals.extend(x_segment_vals)
            y_vals.extend([self._compute_value(self.segment_coeffs[i], x) for x in x_segment_vals])

        return np.array(x_vals), np.array(y_vals)
    
    def integrate_segments(self) -> 'PiecewisePolynomial':
        # Integrates the PiecewisePolynomial. This is done by integrating each segment.
        integrated_poly = PiecewisePolynomial(self.breakpoints, self.segment_coeffs)
        accumulated_y = 0
        for i in range(len(self)):
            start, end = self.breakpoints[i], self.breakpoints[i+1]
            integrated_segment = [0.0] + self.segment_coeffs[i]
            for k in range(1, len(integrated_segment)):
                integrated_segment[k] /= k
            y_start = self._compute_value(integrated_segment, start)
            y_end = self._compute_value(integrated_segment, end)
            integrated_segment[0] = accumulated_y - y_start
            accumulated_y += y_end - y_start
            integrated_poly.segment_coeffs[i] = integrated_segment
        return integrated_poly
    
    @staticmethod
    def _find_extrema(segment, start, end):
        # Finds local extrema within an interval for a polynomial segment.
        if len(segment) <= 1:
            return []  # No extrema possible for a constant or empty polynomial.

        # Compute the derivative of the polynomial to find critical points.
        derivative = [k * coeff for k, coeff in enumerate(segment)][1:]
        roots = np.roots(derivative)
        return [x.real for x in roots if x.imag == 0 and start < x.real < end]

    def get_breakpoints(self):
        # Gets all breakpoints including interval endpoints and local extrema.
        all_breakpoints = set()
        for i in range(len(self)):
            start, end = self.breakpoints[i], self.breakpoints[i + 1]
            all_breakpoints.update([start, end])
            extremum = self._find_extrema(self.segment_coeffs[i], start, end)
            all_breakpoints.update(extremum)
        return sorted(all_breakpoints)
    
    def extreme_value(self, absolute=False):
        # Finds the global minimum and maximum points of the PiecewisePolynomial.
        breakpoints = self.get_breakpoints()
        evaluated_points = [(point, self._compute_value(self.segment_coeffs[self._locate_segment(point)], point)) for point in breakpoints]

        if absolute:
            max_point = max(evaluated_points, key=lambda item: abs(item[1]))
            return max_point

        min_point = min(evaluated_points, key=lambda item: item[1])
        max_point = max(evaluated_points, key=lambda item: item[1])
        return min_point, max_point
