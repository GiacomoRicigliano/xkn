import sys

import numpy as np

labels = {"mass": "mass", "op": "opacity", "vel": "velocity"}


class AngularDistribution(object):
    """
    angular distribution class
    accepts uniform in the angle:           'uniform'
    and uniform in the cosine of the angle: 'cos_uniform'
    """

    def __init__(self, angular_law):
        if angular_law == "uniform":
            self.angular_distribution = self.uniform_ang
        elif angular_law == "cos_uniform":
            self.angular_distribution = self.cos_uniform_ang
        else:
            sys.exit("Unknown angular distribution")

    def __call__(self, n, omega_frac):
        if n not in [6, 9, 12, 15]:
            sys.exit("Error: n_slices must be an even number!")
        return self.angular_distribution(n, omega_frac)

    def uniform_ang(self, n, omega_fraction):
        delta = np.pi / 2.0 / float(n)
        a = np.array([[delta * i, delta * (i + 1)] for i in range(int(n))])
        o = np.array([2.0 * np.pi * (np.cos(x[0]) - np.cos(x[1])) for x in a])
        return a, o * omega_fraction

    def cos_uniform_ang(self, n, omega_fraction):
        delta = 1.0 / float(n)
        a = np.array(
            [
                [np.arccos(delta * float(i)), np.arccos(delta * float(i - 1))]
                for i in range(int(n), 0, -1)
            ]
        )
        o = np.array([2.0 * np.pi * (np.cos(x[0]) - np.cos(x[1])) for x in a])
        return a, o * omega_fraction


class MassAngularDistribution(object):

    def __init__(self, angular_law_distribution):
        self.key = "mass"
        if angular_law_distribution == "uniform":
            self.mass_angular_distribution = uniform_distribution
        elif angular_law_distribution == "sin":
            self.mass_angular_distribution = sin_distribution
        elif angular_law_distribution == "sin2":
            self.mass_angular_distribution = sin2_distribution
        elif angular_law_distribution == "cos2":
            self.mass_angular_distribution = cos2_distribution
        elif angular_law_distribution == "step":
            self.mass_angular_distribution = step_distribution
        else:
            sys.exit("Unknown mass angular distribution")

    def __call__(self, angles, **kwargs):
        return self.mass_angular_distribution(self.key, angles, **kwargs)


class OpacityAngularDistribution(object):

    def __init__(self, angular_law_distribution):
        self.key = "op"
        if angular_law_distribution == "uniform":
            self.opacity_angular_distribution = uniform_distribution
        elif angular_law_distribution == "sin":
            self.opacity_angular_distribution = sin_distribution
        elif angular_law_distribution == "sin2":
            self.opacity_angular_distribution = sin2_distribution
        elif angular_law_distribution == "cos2":
            self.opacity_angular_distribution = cos2_distribution
        elif angular_law_distribution == "abscos":
            self.opacity_angular_distribution = abscos_distribution
        elif angular_law_distribution == "step":
            self.opacity_angular_distribution = step_distribution
        else:
            sys.exit("Unknown opacity angular distribution")

    def __call__(self, angles, **kwargs):
        return self.opacity_angular_distribution(self.key, angles, **kwargs)


class VelocityAngularDistribution(object):

    def __init__(self, angular_law_distribution):
        self.key = "vel"
        if angular_law_distribution == "uniform":
            self.velocity_angular_distribution = uniform_distribution
        elif angular_law_distribution == "sin":
            self.velocity_angular_distribution = sin_distribution
        elif angular_law_distribution == "sin2":
            self.velocity_angular_distribution = sin2_distribution
        elif angular_law_distribution == "cos2":
            self.velocity_angular_distribution = cos2_distribution
        elif angular_law_distribution == "abscos":
            self.velocity_angular_distribution = abscos_distribution
        elif angular_law_distribution == "step":
            self.velocity_angular_distribution = step_distribution
        else:
            sys.exit("Unknown opacity angular distribution")

    def __call__(self, angles, **kwargs):
        return self.velocity_angular_distribution(self.key, angles, **kwargs)


def uniform_distribution(key, angles, **kwargs):
    if key == "mass":
        m_tot = kwargs["m_tot"]
        if m_tot is None:
            sys.exit(
                "Error! user must specify a total {}! exiting\n".format(labels[key])
            )
        return np.array([m_tot * 0.5 * (np.cos(a[0]) - np.cos(a[1])) for a in angles])
    else:
        central_var = kwargs["central_" + key]
        if central_var is None:
            sys.exit("Error! user must specify a {}! exiting\n".format(labels[key]))
        return np.array([central_var for a in angles])


def sin_distribution(key, angles, **kwargs):
    if key == "mass":
        m_tot = kwargs["m_tot"]
        if m_tot is None:
            sys.exit(
                "Error! user must specify a total {}! exiting\n".format(labels[key])
            )
        return np.array(
            [
                (m_tot / np.pi)
                * (
                    a[1]
                    - a[0]
                    - (np.sin(a[1]) * np.cos(a[1]) - np.sin(a[0]) * np.cos(a[0]))
                )
                for a in angles
            ]
        )
    else:
        min_var = kwargs["min_" + key]
        max_var = kwargs["max_" + key]
        if min_var is None:
            sys.exit(
                "Error! user must specify a minimum {}! exiting\n".format(labels[key])
            )
        if max_var is None:
            sys.exit(
                "Error! user must specify a maximum {}! exiting\n".format(labels[key])
            )
        delta_var = max_var - min_var
        return np.array(
            [min_var + delta_var * np.sin(0.5 * (a[1] + a[0])) for a in angles]
        )


def sin2_distribution(key, angles, **kwargs):
    if key == "mass":
        m_tot = kwargs["m_tot"]
        if m_tot is None:
            sys.exit(
                "Error! user must specify a total {}! exiting\n".format(labels[key])
            )
        return np.array(
            [
                m_tot
                * 0.0625
                * (
                    np.cos(3.0 * a[1])
                    - 9.0 * np.cos(a[1])
                    - np.cos(3.0 * a[0])
                    + 9.0 * np.cos(a[0])
                )
                for a in angles
            ]
        )
    else:
        min_var = kwargs["min_" + key]
        max_var = kwargs["max_" + key]
        if min_var is None:
            sys.exit(
                "Error! user must specify a minimum {}! exiting\n".format(labels[key])
            )
        if max_var is None:
            sys.exit(
                "Error! user must specify a maximum {}! exiting\n".format(labels[key])
            )
        delta_var = max_var - min_var
        return np.array(
            [min_var + delta_var * (np.sin(0.5 * (a[1] + a[0])) ** 2) for a in angles]
        )


def cos2_distribution(key, angles, **kwargs):
    if key == "mass":
        m_tot = kwargs["m_tot"]
        if m_tot is None:
            sys.exit(
                "Error! user must specify a total {}! exiting\n".format(labels[key])
            )
        return np.array(
            [m_tot * 0.5 * (np.cos(a[0]) ** 3 - np.cos(a[1]) ** 3) for a in angles]
        )
    else:
        min_var = kwargs["min_" + key]
        max_var = kwargs["max_" + key]
        if min_var is None:
            sys.exit(
                "Error! user must specify a minimum {}! exiting\n".format(labels[key])
            )
        if max_var is None:
            sys.exit(
                "Error! user must specify a maximum {}! exiting\n".format(labels[key])
            )
        delta_var = max_var - min_var
        return np.array(
            [min_var + delta_var * (np.cos(0.5 * (a[1] + a[0])) ** 2) for a in angles]
        )


def abscos_distribution(key, angles, **kwargs):
    if key == "mass":
        sys.exit("Error! abscos angular distribution not defined for mass! exiting\n")
    else:
        min_var = kwargs["min_" + key]
        max_var = kwargs["max_" + key]
        if min_var is None:
            sys.exit(
                "Error! user must specify a minimum {}! exiting\n".format(labels[key])
            )
        if max_var is None:
            sys.exit(
                "Error! user must specify a maximum {}! exiting\n".format(labels[key])
            )
        delta_var = max_var - min_var
        return np.array(
            [min_var + delta_var * (abs(np.cos(0.5 * (a[1] + a[0])))) for a in angles]
        )


def step_distribution(key, angles, **kwargs):
    if key == "mass":
        m_tot = kwargs["m_tot"]
        if m_tot is None:
            sys.exit(
                "Error! user must specify a total {}! exiting\n".format(labels[key])
            )
        step_angle = kwargs["step_angle_mass"]
        high_lat_flag = kwargs["high_lat_flag"]
        if step_angle is None:
            sys.exit("Error! Must specify a step angle in radians! exiting\n")
        if high_lat_flag is None:
            sys.exit("Error! User must specify high or low latitude side! exiting\n")
        elif high_lat_flag != 1 and high_lat_flag != 0:
            sys.exit(
                "Error! User must specify 1 for high latitude and 0 for low latitude! exiting\n"
            )

        prefac1 = m_tot * 0.5 / (1.0 - np.cos(step_angle))
        prefac2 = m_tot * 0.5 / np.cos(step_angle)
        maxval = np.maximum(m_tot * 1.0e-4, 1.0e-5)
        if high_lat_flag:
            return np.array(
                [
                    (
                        prefac1 * (np.cos(a[0]) - np.cos(a[1]))
                        if (np.sin(0.5 * (a[1] + a[0])) < np.sin(step_angle))
                        else maxval
                    )
                    for a in angles
                ]
            )
        else:
            return np.array(
                [
                    (
                        prefac2 * (np.cos(a[0]) - np.cos(a[1]))
                        if (np.sin(0.5 * (a[1] + a[0])) > np.sin(step_angle))
                        else maxval
                    )
                    for a in angles
                ]
            )
    else:
        step_angle_var = kwargs["step_angle_" + key]
        high_lat_var = kwargs["high_lat_" + key]
        low_lat_var = kwargs["low_lat_" + key]
        if step_angle_var is None:
            sys.exit("Error! Must specify a step angle in radians! exiting\n")
        if high_lat_var is None:
            sys.exit(
                "Error! user must specify a high latitude {}! exiting\n".format(
                    labels[key]
                )
            )
        if low_lat_var is None:
            sys.exit(
                "Error! user must specify a low latitude {}! exiting\n".format(
                    labels[key]
                )
            )
        return np.array(
            [
                (
                    high_lat_var
                    if (np.sin(0.5 * (a[1] + a[0])) < np.sin(step_angle_var))
                    else low_lat_var
                )
                for a in angles
            ]
        )
