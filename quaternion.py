import numpy as np


class Quaternion:

    # A 4x1 numpy array where q = [vector, scalor]
    def __init__(self, q):
        if q.shape == (4, 1):
            self._q = q
        else:
            raise ValueError("Quaternion shape must be (4,1)")

    @property
    def value(self):
        return self._q

    @value.setter
    def value(self, new_value):
        self._q = new_value

    @property
    def vector(self):
        return self._q[0:3].reshape(3, 1)

    @vector.setter
    def vector(self, new_value):
        if new_value.shape == (3, 1):
            self._q[0:3] = new_value
        else:
            raise ValueError("Vector shape must be (3,1)")

    @property
    def scalor(self):
        return self._q[-1][0]

    @scalor.setter
    def scalor(self, new_value):
        self._q[-1][0] = new_value

    @property
    def q1(self):
        return self._q[0][0]

    @q1.setter
    def q1(self, new_value):
        self._q[0][0] = new_value

    @property
    def q2(self):
        return self._q[1][0]

    @q2.setter
    def q2(self, new_value):
        self._q[1][0] = new_value

    @property
    def q3(self):
        return self._q[2][0]

    @q3.setter
    def q3(self, new_value):
        return self._q[2][0]

    @property
    def q4(self):
        return self._q[3][0]

    @q4.setter
    def q4(self, new_value):
        self._q[3][0] = new_value

    def __str__(self):
        return str(self.value.flatten())

    def __repr__(self):
        return

    def norm(self):
        return np.linalg.norm(self.value)

    def normalized(self):
        return Quaternion(self.value/np.linalg.norm(self.value))

    def conjugate(self):
        return Quaternion(np.array([-self.q1, -self.q2, -self.q3, self.q4]).reshape(4, 1))

    def inverse(self):
        q_mag = self.norm()
        q_conj = self.conjugate()
        return Quaternion(q_conj.value / (q_mag**2))

    @staticmethod
    def identity():
        return Quaternion(np.array([0, 0, 0, 1]).reshape(4, 1))

    @staticmethod
    def phi(q):
        if not isinstance(q, Quaternion):
            raise TypeError("Expected Quaternion")

        q1, q2, q3, q4 = q.q1, q.q2, q.q3, q.q4

        return np.array([
            [q4,  q3, -q2],
            [-q3,  q4,  q1],
            [q2, -q1,  q4],
            [-q1, -q2, -q3]
        ])

    @staticmethod
    def eps(q):
        if not isinstance(q, Quaternion):
            raise TypeError("Expected Quaternion")

        q1, q2, q3, q4 = q.q1, q.q2, q.q3, q.q4

        return np.array([
            [q4,  -q3, q2],
            [q3,  q4,  -q1],
            [-q2, q1,  q4],
            [-q1, -q2, -q3]
        ])

    def cross(self, q2):
        result = np.block([Quaternion.phi(self), self.value]) @ q2.value
        return Quaternion(np.array([result[0], result[1], result[2], result[3]]).reshape(4, 1))

    def dot(self, q2):
        result = np.block([Quaternion.eps(self), self.value]) @ q2.value
        return Quaternion(np.array([result[0], result[1], result[2], result[3]]).reshape(4, 1))

    def kinamatics(self, w):
        return Quaternion(0.5*Quaternion.eps(self) @ w)
