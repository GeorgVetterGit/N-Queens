import numpy as np
import pygame
import asyncio


# Calculate GA History
N = 12
POPULATION_SIZE = 100
RECOMB_RATE = 0.1
MUT_RATE = 0.05
GENERATIONS = 1000

recombinations = int(POPULATION_SIZE * RECOMB_RATE)

def fitness(population_list: list[list])->list:
    fitness_list = []
    for individuum in population_list:
        collision = 0
        for i in range(N):
            for j in range(i + 1, N):
                distance = abs(i - j)
                start_pos = individuum[i]
                comp_pos = individuum[j]
                if (comp_pos + distance == start_pos):
                    collision += 1
                if (comp_pos - distance == start_pos):
                    collision += 1
                if comp_pos == start_pos:
                    collision += 1
        fitness_list.append(collision)
    return fitness_list

def recombination(population_list: list[list], fitness_list: list)->list[list]:
    sum_fitness = sum(fitness_list)
    prop_fitness = [fitness / sum_fitness for fitness in fitness_list]
    roul_wheel = [sum(prop_fitness[:i]) for i in range(1,POPULATION_SIZE)] + [1]
    parents = []
    for _ in range(2):
        pointer = np.random.random()
        for idx in range(POPULATION_SIZE):
            if roul_wheel[idx] > pointer:
                parents.append(idx)
                break
    crossover_point = np.random.choice(list(range(1, N + 1)))
    population_list.append(np.array(list(population_list[parents[0]][:crossover_point]) + list(population_list[parents[1]][crossover_point:])))
    population_list.append(np.array(list(population_list[parents[1]][:crossover_point]) + list(population_list[parents[0]][crossover_point:])))
    return population_list

def mutation(population_list: list[list])->list[list]:
    for idx, individuum in enumerate(population_list):
        prob = np.random.random()
        if prob <= MUT_RATE:
            population_list.pop(idx)
            allel = np.random.choice(list(range(N)))
            individuum[allel] = np.random.choice(list(range(1,N+1)))
            population_list.append(individuum)
    return population_list

def selection(population_list: list[list], fitness_list: list)->list[list]:
    pop_idxs = np.argsort(fitness_list)[:POPULATION_SIZE]
    population_list = [population_list[idx] for idx in pop_idxs]
    return population_list

def ga_main():
    population_list = [np.random.randint(1,N+1,size=N) for i in range(POPULATION_SIZE)]
    history_list = []
    for generation in range(GENERATIONS):
        fitness_list = fitness(population_list)
        history_list.append((generation, population_list[list(np.argsort(fitness_list))[0]]))
        if min(fitness_list) == 0:
            break
        for _ in range(recombinations):
            population_list = recombination(population_list, fitness_list)
        population_list = mutation(population_list)
        fitness_list = fitness(population_list)
        population_list = selection(population_list, fitness_list)
    return history_list

history_list = ga_main()

### Game

pygame.init()

COLORS = {'BLACK': (40, 40, 40),
         'WHITE': (255, 255, 255),
         'RED': (255, 0, 0)}

WIDTH = 800
HEIGTH = 800
box_size = 800 / N
SIDEBAR = 200

screen = pygame.display.set_mode((WIDTH + SIDEBAR, HEIGTH))
clock = pygame.time.Clock()

pygame.display.set_caption(f"{str(N)}-Queens Optimization via Genetic Algorithm")

PATH_QUEEN = 'queen.png'
IMG_QUEEN = pygame.image.load(PATH_QUEEN).convert_alpha()
IMG_QUEEN = pygame.transform.scale(IMG_QUEEN,(box_size * 0.6, box_size * 0.6))

queens = pygame.sprite.Group()

class queen(pygame.sprite.Sprite):

    def __init__(self, row, groups):
        super().__init__(groups)
        self.pos = 1
        self.row = row
        self.start = 1
        self.image = IMG_QUEEN
        self.rect = self.image.get_rect(center = (self.row * box_size - (box_size / 2), (800 + box_size) - self.pos * box_size - (box_size / 2)))
        self.counter = 0

    def update(self, destinations):
        if self.start == self.pos:
            self.counter = 0
        distance = destinations[self.row - 1] - self.start
        step = distance / 30
        if self.counter < 30:
            self.pos += step
            self.counter += 1
        else:
            self.start = self.pos
        self.rect = self.image.get_rect(center = (self.row * box_size - (box_size / 2), (800 + box_size) - self.pos * box_size - (box_size / 2)))

def text_center(screen: 'screen', text_string: str, x: int, y: int, font_size: int)->'screen':
    font = pygame.font.Font('freesansbold.ttf', font_size)
    text = font.render(text_string,
                        True,
                        COLORS['BLACK'],
                        COLORS['WHITE'],
                        )
    textRect = text.get_rect()
    textRect.center = (x, y)
    screen.blit(text, textRect)
    return screen

queen_list = []

for i in range(N):
    queen_list.append(queen(i + 1, queens))

time = 0
running = True

async def main():
    global N, POPULATION_SIZE, RECOMB_RATE, MUT_RATE, GENERATIONS, recombinations, history_list, COLORS, WIDTH, HEIGTH, box_size, SIDEBAR, screen, clock, PATH_QUEEN, IMG_QUEEN, IMG_QUEEN, queens, queen_list, time, running
    while running:
        clock.tick(60)

        time += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: #restart
                    history_list = ga_main()
                    time = 0
                    queens = pygame.sprite.Group()
                    queen_list = []
                    for i in range(N):
                        queen_list.append(queen(i + 1, queens))

        screen.fill(COLORS['WHITE'])

        for col in range(N):
            for row in range(N):
                if col % 2:
                    if row % 2:
                        color = 'BLACK'
                    else:
                        color = 'WHITE'
                else:
                    if row % 2:
                        color = 'WHITE'
                    else:
                        color = 'BLACK'
                pygame.draw.rect(screen, COLORS[color], pygame.Rect(col * box_size, row * box_size,box_size,box_size))

        idx = time // 20

        if idx < len(history_list):
            individuum = history_list[idx][1]
            generation = idx
            N = len(individuum)
            new_collision_list = []
            collisions = 0
            for i in range(N):
                for j in range(i + 1, N):
                    distance = abs(i - j)
                    start_pos = individuum[i]
                    comp_pos = individuum[j]
                    if (comp_pos + distance == start_pos):
                        new_collision_list.append((i,j))
                        collisions += 1
                    if (comp_pos - distance == start_pos):
                        new_collision_list.append((i,j))
                        collisions += 1
                    if comp_pos == start_pos:
                        new_collision_list.append((i,j))
                        collisions += 1
            new_collision_list.sort()
            dist = box_size
            for entry in new_collision_list:
                start = (queen_list[entry[0]].rect[0] + queen_list[entry[0]].rect[2] / 2, queen_list[entry[0]].rect[1] + queen_list[entry[0]].rect[3] / 2)
                end = (queen_list[entry[1]].rect[0] + queen_list[entry[0]].rect[2] / 2, queen_list[entry[1]].rect[1] + queen_list[entry[0]].rect[3] / 2)
                pygame.draw.line(screen, COLORS['RED'], start, end, width=10)
        else:
            screen = text_center(screen, 'FINISHED!', WIDTH + 100, 650, 25)

        queens.update(individuum)
        queens.draw(screen)

        screen = text_center(screen, 'STATS', WIDTH + 100, 50, 35) 

        screen = text_center(screen, 'Generation:', WIDTH + 100, 100, 25)
        screen = text_center(screen, str(generation) + '/' + str(len(history_list) - 1), WIDTH + 100, 150, 25)

        pygame.draw.line(screen, COLORS['BLACK'], (WIDTH + 50, 200), (WIDTH + SIDEBAR - 50, 200))

        screen = text_center(screen, 'Collisions:', WIDTH + 100, 250, 25)
        screen = text_center(screen, str(collisions), WIDTH + 100, 300, 25)

        pygame.draw.line(screen, COLORS['BLACK'], (WIDTH + 50, 350), (WIDTH + SIDEBAR - 50, 350))

        screen = text_center(screen, 'Pop.-size:', WIDTH + 100, 400, 20)
        screen = text_center(screen, str(POPULATION_SIZE), WIDTH + 100, 430, 15)
        screen = text_center(screen, 'Recomb Rate:', WIDTH + 100, 480, 20)
        screen = text_center(screen, str(RECOMB_RATE), WIDTH + 100, 510, 15)
        screen = text_center(screen, 'Mut.-Rate:', WIDTH + 100, 560, 20)
        screen = text_center(screen, str(MUT_RATE), WIDTH + 100, 590, 15)

        screen = text_center(screen, '(r) for restart', WIDTH + 100, 750, 15)

        pygame.display.update()

        await asyncio.sleep(0)

asyncio.run(main())