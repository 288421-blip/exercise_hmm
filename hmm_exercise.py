# PLAIN FUNCTIONS 
import math
import json
import itertools
import os
from Bio import SeqIO


def read_model(path):
    f = open(path)
    data = json.load(f)
    f.close()
    return HMM(data["states"], data["alphabet"], data["pi"], data["a"], data["e"])

def log2(p):
    if p == 0:
        value = -math.inf
    else:
        value = math.log2(p)
    return value

print(log2(73))   # 6,189825
print(log2(0))    # -math.inf

def log_sum(a,b):
    if a == -math.inf:
        return float(b)
    elif b == -math.inf:
        return float(a)
    else:
        return float(max(a,b)+log2(1+2**(min(a,b)-max(a,b))))


def sum_values(table):
    total = 0.0
    for key in table:
        total += table[key]
    return total

def same_keys(table, items):   # table es un dict cuyas claves son los items (lista)
    if len(table) != len(items):
        return False
    for item in items:
        if item not in table:
            return False
    for key in table:
        if key not in items:
            return False
    return True


def agreement(z1, z2):
    if len(z1) != len(z2):
        raise ValueError("paths must have the same length")
    if len(z1) == 0:
        return 1.0
    same = 0
    for i in range(len(z1)):
        if z1[i] == z2[i]:
            same += 1
    return same / len(z1)



class HMM:
    def __init__(self, states, alphabet, pi, a, e):
        self.states = states
        self.alphabet = alphabet
        self.pi = pi
        self.a = a
        self.e = e

        self.check_model()

        # Tablas en log2
        self.log_pi = {}
        for s in self.states:
            self.log_pi[s] = log2(self.pi[s])

        self.log_a = {}
        for s in self.states:
            self.log_a[s] = {}
            for t in self.states:
                self.log_a[s][t] = log2(self.a[s][t])

        self.log_e = {}
        for s in self.states:
            self.log_e[s] = {}
            for letter in self.alphabet:
                self.log_e[s][letter] = log2(self.e[s][letter])


    def check_model(self):
        if not same_keys(self.pi, self.states):
            raise ValueError("pi must have exactly the keys of states")
        if not same_keys(self.a, self.states):
            raise ValueError("a must have exactly the keys of states")
        if not same_keys(self.e, self.states):
            raise ValueError("e must have exactly the keys of states")

        for s in self.states:
            if not same_keys(self.a[s], self.states):
                raise ValueError("row of a for state " + str(s) + " must have exactly the keys of states")
            if not same_keys(self.e[s], self.alphabet):
                raise ValueError("row of e for state " + str(s) + " must have exactly the letters of alphabet")

        if not math.isclose(sum_values(self.pi), 1.0):
            raise ValueError("pi must sum to 1")

        for s in self.states:
            if not math.isclose(sum_values(self.a[s]), 1.0):
                raise ValueError("row of a for state " + str(s) + " must sum to 1")
            if not math.isclose(sum_values(self.e[s]), 1.0):
                raise ValueError("row of e for state " + str(s) + " must sum to 1")

    def joint_probability(self, x, z):
        if len(x) != len(z):
            raise ValueError("x and z must have the same exact length")
        if len(x) == 0:
            return 1.0
        p = self.pi[z[0]] * self.e[z[0]][x[0]]
        for i in range(1, len(x)):
            p = p * self.a[z[i-1]][z[i]] * self.e[z[i]][x[i]]
        return p

    def log_joint(self, x, z):
        if len(x) != len(z):
            raise ValueError("x and z must have the same exact length")
        if len(x) == 0:
            return 0.0
        total = self.log_pi[z[0]] + self.log_e[z[0]][x[0]]
        for i in range(1, len(x)):
            total = total + self.log_a[z[i-1]][z[i]] + self.log_e[z[i]][x[i]]
        return total

    def brute_force(self, x):
        best_path = ""
        best_prob = -1.0
        total = 0.0
        for path in itertools.product(self.states, repeat=len(x)):
            z = "".join(path)
            p = self.joint_probability(x, z)
            total = total + p
            if p > best_prob:
                best_prob = p
                best_path = z

        return (best_path, best_prob, total)

# TASK 2 ##############################################################
    def viterbi(self, x):
        n = len(x)
        if n == 0:
            return ("", 0.0)

        v = []
        back = []

        # Columna 0: solo cuenta empezar en k y emitir x[0]
        v.append({})
        back.append({})
        for k in self.states:
            v[0][k] = self.log_pi[k] + self.log_e[k][x[0]]
            back[0][k] = None

        # Columnas 1 a n-1
        for i in range(1, n):
            v.append({})
            back.append({})
            for k in self.states:
                best_score = -math.inf
                best_prev = self.states[0]
                for j in self.states:
                    score = v[i-1][j] + self.log_a[j][k]
                    if score > best_score:
                        best_score = score
                        best_prev = j
                v[i][k] = best_score + self.log_e[k][x[i]]
                back[i][k] = best_prev

        # Mejor estado de la última columna
        last = self.states[0]
        best_score = v[n - 1][last]
        for k in self.states:
            if v[n-1][k] > best_score:
                best_score = v[n-1][k]
                last = k

        # Traceback: seguir back hacia atrás y dar la vuelta
        path_list = [last]
        state = last
        for i in range(n-1, 0, -1):
            state = back[i][state]
            path_list.append(state)
        path_list.reverse()

        return ("".join(path_list), best_score)

    # Think about it: lo mismo pero multiplicando probabilidades normales en lugar de añadir logaritmos
    def viterbi_plain(self, x):
        n = len(x)
        if n == 0:
            return ("", 1.0)

        v = []
        back = []

        v.append({})
        back.append({})
        for k in self.states:
            v[0][k] = self.pi[k] * self.e[k][x[0]]
            back[0][k] = None

        for i in range(1, n):
            v.append({})
            back.append({})
            for k in self.states:
                best_score = -1.0
                best_prev = self.states[0]
                for j in self.states:
                    score = v[i-1][j] * self.a[j][k]
                    if score > best_score:
                        best_score = score
                        best_prev = j
                v[i][k] = best_score * self.e[k][x[i]]
                back[i][k] = best_prev

        last = self.states[0]
        best_score = v[n - 1][last]
        for k in self.states:
            if v[n-1][k] > best_score:
                best_score = v[n-1][k]
                last = k

        path_list = [last]
        state = last
        for i in range(n-1, 0, -1):
            state = back[i][state]
            path_list.append(state)
        path_list.reverse()

        return ("".join(path_list), best_score)


# TASK 3 ##########################################################
    def forward(self, x):
        n = len(x)
        if n == 0:
            return ([], 0.0)   # P("") = 1  ->  log2 = 0

        f = []

        # Columna 0
        f.append({})
        for k in self.states:
            f[0][k] = self.log_pi[k] + self.log_e[k][x[0]]

        # Columnas 1 a n-1: suma sobre todos los estados previos j
        for i in range(1, n):
            f.append({})
            for k in self.states:
                total = -math.inf
                for j in self.states:
                    total = log_sum(total, f[i-1][j] + self.log_a[j][k])
                f[i][k] = total + self.log_e[k][x[i]]

        # log2 P(x) = log_sum sobre la última columna
        log_px = -math.inf
        for k in self.states:
            log_px = log_sum(log_px, f[n-1][k])

        return (f, log_px)


    def backward(self, x):
        n = len(x)
        if n == 0:
            return ([], 0.0)   # P("") = 1  ->  log2 = 0

        # Inicializamos tabla con n columnas
        b = [None] * n

        # Última columna: log2(1) = 0 para todos los estados
        b[n-1] = {}
        for k in self.states:
            b[n-1][k] = 0.0

        # Columnas n-2 hasta 0
        for i in range(n-2, -1, -1):
            b[i] = {}
            for k in self.states:
                total = -math.inf
                for l in self.states:
                    term = self.log_a[k][l] + self.log_e[l][x[i+1]] + b[i+1][l]
                    total = log_sum(total, term)
                b[i][k] = total

        # log2 P(x) = log_sum sobre k de log_pi[k] + log_e[k][x[0]] + b[0][k]
        log_px = -math.inf
        for k in self.states:
            term = self.log_pi[k] + self.log_e[k][x[0]] + b[0][k]
            log_px = log_sum(log_px, term)

        return (b, log_px)


    def posterior(self, x):
        n = len(x)
        if n == 0:
            return [None]

        f, log_px = self.forward(x)
        b, _ = self.backward(x)

        post = []
        for i in range(n):
            post.append({})
            for k in self.states:
                # P(z_i = k | x) = 2^(f + b - log2 P(x))
                post[i][k] = 2 ** (f[i][k] + b[i][k] - log_px)

        return post


    def posterior_path(self, x):
        post = self.posterior(x)

        path_list = []
        for i in range(len(post)):
            best_state = self.states[0]
            best_prob = post[i][best_state]
            for k in self.states:
                if post[i][k] > best_prob:    # estricto: en empate gana el primero
                    best_prob = post[i][k]
                    best_state = k
            path_list.append(best_state)

        return "".join(path_list)



# SELF CHECKS OF TASKS 2 AND 3
if __name__ == "__main__":
    folder = os.path.dirname(os.path.abspath(__file__))

    toy = read_model(os.path.join(folder, "toy_model.json"))
    weather = read_model(os.path.join(folder, "weather_model.json"))

    # TASK 2
    for m in [toy, weather]:
        for n in range(1, 7):
            for letters in itertools.product(m.alphabet, repeat=n):
                x = "".join(letters)
                path, score = m.viterbi(x)
                bf_path, bf_prob, bf_total = m.brute_force(x)

                assert path == bf_path
                assert math.isclose(2 ** score, bf_prob)
                assert math.isclose(score, m.log_joint(x, path))
    print("viterbi: self-checks OK")   # sale: viterbi: self-checks OK  -> checked

    print(weather.viterbi("ssrssrrrss"))   # ('HHHHHLLLHH', -14.14...)  -> checked
    print(toy.viterbi("phchc"))            # ('SMMMS', -11.89...)  -> checked (en papel -11.87 por el redondeo)

    # Think about it 
    for record in SeqIO.parse(os.path.join(folder, "sequences.fasta"), "fasta"):
        x = str(record.seq)
        path_log, score = toy.viterbi(x)
        path_plain, prob = toy.viterbi_plain(x)
        print(record.id, len(x), score / len(x), prob, path_log == path_plain)

        # From the length of 1000, the plain version breakes as it returns a probability of 0.0 and the path list of
        # plain version is not longer equal to the original one (the logaritmic). That's also the reason why it's not reliable.

        # The best path costs roughly 1.75 to 1.8 bits per position (score/length is almost constant: -1.77 at 300, -1.79 at 600, 
        # -1.78 at 1000, -1.75 at 2000, -1.74 at 5000). So the log2 score grows linearly with the length, while the plain probability does it exponentially 
        # (about 2^-1.75 per position).

    # TASK 3
    for m in [toy, weather]:
        for n in range(1, 7):
            for letters in itertools.product(m.alphabet, repeat=n):
                x = "".join(letters)

                f, log_f = m.forward(x)
                b, log_b = m.backward(x)
                post = m.posterior(x)
                bf_path, bf_prob, bf_total = m.brute_force(x)

                # 1. forward y backward dan el mismo log2 P(x)
                assert math.isclose(log_f, log_b)

                # 2. 2 ** log2 P(x) coincide con la suma de fuerza bruta
                assert math.isclose(2 ** log_f, bf_total)

                # 3. cada columna de posterior suma 1
                for i in range(n):
                    assert math.isclose(sum(post[i].values()), 1.0)

                # 4. (opcional) posterior contra fuerza bruta
                num = {}
                for i in range(n):
                    num[i] = {}
                    for k in m.states:
                        num[i][k] = 0.0
                for path in itertools.product(m.states, repeat=n):
                    z = "".join(path)
                    p = m.joint_probability(x, z)
                    for i in range(n):
                        num[i][z[i]] += p
                for i in range(n):
                    for k in m.states:
                        assert math.isclose(post[i][k], num[i][k] / bf_total)

    print("forward/backward/posterior: self-checks OK")

    # 5. Secuencia del handout
    x = "phchc"
    f, log_px = toy.forward(x)
    b, _ = toy.backward(x)
    post = toy.posterior(x)
    for i in range(len(x)):
        for k in toy.states:
            print(i, k,
                  "f =", 2 ** f[i][k],
                  "b =", 2 ** b[i][k],
                  "post =", post[i][k])
    print("P(x) =", 2 ** log_px)