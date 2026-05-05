    def z_score(self, x):
        """
        Calculates the z-score of a given x-value

        Parameters:
        x (float): x-value

        Returns:
        float: z-score
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score

        Parameters:
        z (float): z-score

        Returns:
        float: x-value
        """
        return self.mean + z * self.stddev
