import numpy as np
import matplotlib.pyplot as plt

DNA_size = 10            # DNA length
Pop_size = 100           # population size
crossover_rate = 0.8         # mating probability (DNA crossover)
mutation_rate = 0.008    # mutation probability
N_generations = 500
X_BOUND = [0, 5]         # x upper and lower bounds

# to find the maximum of the equation below
def F(x):
    return np.sin(10*x)*x + np.cos(2*x)*x

# find non-zero fitness for selection since sol might be negative
def get_fitness(pred):
    return pred + 1e-3 - np.min(pred)

# makes genotype to phenotype [::-1],::-為把張量內的矩陣或向量到序排列，排列到第幾個分量
# convert binary DNA to decimal and normalize it to a range(0, 5)
def translateDNA(pop):
    return pop.dot(2 ** np.arange(DNA_size)[::-1]) / float(2**DNA_size-1) * X_BOUND[1]

# nature selection wrt pop's fitness
# but the lower fitness would not be kill
def select(pop, fitness):
    idx = np.random.choice(np.arange(Pop_size), size=Pop_size, replace=True,
                           p = fitness/fitness.sum())
    return pop[idx]

def crossover(parent, pop):
    if np.random.rand () < crossover_rate:
        i_ = np.random.randint (0, Pop_size, size=1)  # select another individual from pop
        cross_points = np.random.randint (0, 2, size=DNA_size).astype (np.bool)  # choose crossover points
        parent [cross_points] = pop [i_, cross_points]  # mating and produce one child
    return parent


def mutate(child):
    for point in range(DNA_size):
        if np.random.rand() < mutation_rate:
            child[point] = 1 if child[point] == 0 else 0
    return child

pop = np.random.randint(2, size=(Pop_size, DNA_size)) #init Pop DNA

# something about plotting
# 打開交互模式
plt.ion()
x = np.linspace(*X_BOUND, 200) # 產生解空間(離散的50個點)
plt.plot(x, F(x))

for _ in range(N_generations):
    F_values = F(translateDNA(pop))

    # something about plotting
    if 'sca' in globals (): sca.remove ()
    sca = plt.scatter (translateDNA (pop), F_values, s=200, lw=0, c='red', alpha=0.5);
    plt.pause (0.05)

    # GA part (evolution)
    fitness = get_fitness(F_values)
    print("Most fitted DNA: ", pop[np.argmax(fitness), :]) # 把每個DNA最優的基因都列出來
    pop = select(pop, fitness)
    pop_copy = pop.copy() # 把較劣勢的基因保留，可能會產生更好的基因
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child       # parent is replaced by its child

plt.ioff(); plt.show()