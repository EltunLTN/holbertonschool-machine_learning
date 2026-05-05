    def pmf(self, k):
        """
        Calculates the PMF for a given number of successes

        Args:
            k (int): number of successes

        Returns:
            float: PMF value
        """

        k = int(k)

        if k < 0 or k > self.n:
            return 0

        # factorial helper
        def factorial(x):
            result = 1
            for i in range(1, x + 1):
                result *= i
            return result

        # combinations nCk
        comb = factorial(self.n) / (factorial(k) * factorial(self.n - k))

        return comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))
