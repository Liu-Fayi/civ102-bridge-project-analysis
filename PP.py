from copy import deepcopy
import numpy as np

class PiecewisePolynomial():
    def __init__(self, keypoints, pieces):
        self.num_pieces = len(keypoints) - 1
        self.keypoints = deepcopy(sorted(keypoints))
        self.pieces = deepcopy(pieces)

    def __len__(self):
        return self.num_pieces

    @staticmethod
    def _eval_piece(coefficients, x):
        return np.polyval(coefficients[::-1], x)

    def _find_interval(self, x):
        low, high = 0, len(self.keypoints) - 2  # Adjust the upper bound to len(keypoints) - 2
        while low <= high:
            mid = (low + high) // 2
            if self.keypoints[mid] <= x < self.keypoints[mid + 1]:
                return mid  # Return the index of the interval
            elif x < self.keypoints[mid]:
                high = mid - 1
            else:
                low = mid + 1
        return -1  # Indicate that the value is not within any interval


    def eval(self, x):
        index = self._find_interval(x)
        if index is not None:
            return self._eval_piece(self.pieces[index], x)
        return None

    def evals(self, x_variables):
        return np.array([self.eval(x) for x in x_variables])
    
    def get_plot_points(self, num_splits = 1000):
        """Generate a list of points for plotting the piecewise polynomial."""
        total_length = self.keypoints[-1] - self.keypoints[0]
        xs, ys = [], []

        for i in range(len(self)):
            x1, x2 = self.keypoints[i], self.keypoints[i+1]
            interval_length = x2 - x1
            n_dif = int(np.round(num_splits * (interval_length / total_length)))

            # Generate x-values for this piece
            # Make sure the end point is included 
            x_values = np.linspace(x1, x2, n_dif, endpoint=(i == len(self) - 1))
            xs.extend(x_values)

            # Evaluate the polynomial piece at each x-value
            ys.extend([self._eval_piece(self.pieces[i], x) for x in x_values])

        return np.array(xs), np.array(ys)
    
    def integrate(self) -> 'PiecewisePolynomial':
        """zero vertical displacement at left endpoint"""
        poly = PiecewisePolynomial(self.keypoints, self.pieces)
        sum_y = 0
        for i in range(len(self)):
            x1, x2 = self.keypoints[i], self.keypoints[i+1]
            piece = [0.0] + self.pieces[i]
            for k in range(1, len(piece)):
                piece[k] /= k
            y1 = self._eval_piece(piece, x1)
            y2 = self._eval_piece(piece, x2)
            piece[0] = sum_y - y1
            sum_y += y2 - y1
            poly.pieces[i] = piece
        return poly
    
    @staticmethod
    def _piece_optim(piece, x1, x2):
        """Find optimal x values (local extrema) within a given interval for a polynomial piece."""
        if len(piece) <= 1:
            return []  # No extrema possible for a constant or empty polynomial

        # Compute the derivative of the polynomial
        derivative = [k * coeff for k, coeff in enumerate(piece)][1:]

        # Find roots of the derivative
        roots = np.roots(derivative)

        # Filter to include only real roots within the interval [x1, x2]
        return [x.real for x in roots if x.imag == 0 and x1 < x.real < x2]

    def get_keypoints(self):
        """Get all keypoints including interval endpoints and local extrema."""
        all_keypoints = set()

        for i in range(len(self)):
            x1, x2 = self.keypoints[i], self.keypoints[i + 1]
            
            # Add interval endpoints
            all_keypoints.update([x1, x2])

            # Find and add extreme value within the interval
            extreme_val = self._piece_optim(self.pieces[i], x1, x2)
            all_keypoints.update(extreme_val)
        return sorted(all_keypoints)
    
    
    def get_optim(self, absolute=False):
        """Find the global minimum and maximum points of the piecewise polynomial."""
        keypoints = self.get_keypoints()
        evaluated_points = [(x, self._eval_piece(self.pieces[self._find_interval(x)], x)) for x in keypoints]

        if absolute:
            # Find the point with the maximum absolute value
            max_point = max(evaluated_points, key=lambda item: abs(item[1]))
            return max_point

        # Find the global minimum and maximum
        min_point = min(evaluated_points, key=lambda item: item[1])
        max_point = max(evaluated_points, key=lambda item: item[1])

        return min_point, max_point
    
    def mul(self, c) -> 'PiecewisePolynomial':
        """multiply by a constant"""
        poly = PiecewisePolynomial(self.keypoints, self.pieces)
        for i in range(len(poly.pieces)):
            piece = self.pieces[i][:]
            for k in range(len(piece)):
                piece[k] *= c
            poly.pieces[i] = piece
        return poly

    @staticmethod
    def _sub_piece(piece1, piece2):
        res = [0] * max(len(piece1), len(piece2))
        for i in range(len(piece1)):
            res[i] += piece1[i]
        for i in range(len(piece2)):
            res[i] -= piece2[i]
        return res
    
    def sub(self, piece) -> 'PiecewisePolynomial':
        """Subtract a polynomial from each piece of the piecewise polynomial."""
        new_pieces = [self._sub_piece(p, piece) for p in self.pieces]
        return PiecewisePolynomial(self.keypoints, new_pieces)
    
    def _mul_piece(p, q):
        """Multiply two polynomials."""
        result = [0] * (len(p) + len(q) - 1)
        for i, coeff_p in enumerate(p):
            for j, coeff_q in enumerate(q):
                result[i + j] += coeff_p * coeff_q
        return 

    def polymul(self, that) -> 'PiecewisePolynomial':
        """Multiply two piecewise polynomials."""
        # Combine and sort the unique keypoints from both polynomials
        combined_keypoints = sorted(set(self.keypoints + that.keypoints))

        new_pieces = []
        for i in range(len(combined_keypoints) - 1):
            # Find the current interval
            interval_start, interval_end = combined_keypoints[i], combined_keypoints[i + 1]

            # Find corresponding pieces for the current interval in both polynomials
            piece_this = self._get_piece_at(self, interval_start)
            piece_that = self._get_piece_at(that, interval_start)

            # Multiply the pieces and add to the new pieces
            new_pieces.append(self._mul_piece(piece_this, piece_that))

        return PiecewisePolynomial(combined_keypoints, new_pieces)

    @staticmethod
    def _get_piece_at(poly, x):
        """Get the polynomial piece applicable at the given x in the piecewise polynomial."""
        for i in range(len(poly.keypoints) - 1):
            if poly.keypoints[i] <= x < poly.keypoints[i + 1]:
                return poly.pieces[i]
        return [0]  # Return a zero polynomial if x is outside the range
