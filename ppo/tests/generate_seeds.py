import random

def generate_seeds(filename="seeds.txt", count=10, seed_range=(1, 1_000_000)):
    seeds = set()
    while len(seeds) < count:
        seeds.add(random.randint(*seed_range))

    seed_list = list(seeds)
    random.shuffle(seed_list)  # 무작위 순서로 섞기

    with open(filename, "w") as f:
        for seed in seed_list:
            f.write(f"{seed}\n")

    print(f"{count} unique seeds saved to {filename}")

if __name__ == "__main__":
    # 필요에 따라 count를 조절하세요
    generate_seeds(filename="./ppo/tests/validation_seeds.txt", count=1000)
