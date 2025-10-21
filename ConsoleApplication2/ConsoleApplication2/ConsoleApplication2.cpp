#include <mpi.h>
#include <stdio.h>
#include <vector>
#include <numeric> // Для std::accumulate

/**
 * @brief Вирішує Завдання 1:
 * 1. Створює довгий вектор у кореневому (root) процесі.
 * 2. Розподіляє (Scatter) рівні частини вектора на всі процеси[cite: 19, 156, 159].
 * 3. Кожен процес обчислює локальну часткову суму.
 * 4. Збирає (Reduce) всі часткові суми в кореневому процесі,
 * використовуючи колективну обчислювальну операцію MPI_SUM[cite: 23, 253, 263, 287].
 */
void solve_problem_1(int rank, int numtasks) {
    const int ROOT_PROCESS = 0;
    const int ELEMENTS_PER_PROC = 10; // Кількість елементів для кожного процесу
    int vector_size = numtasks * ELEMENTS_PER_PROC;

    // 'long_vector' використовується тільки кореневим процесом
    std::vector<int> long_vector;
    // 'sub_vector' використовується кожним процесом для отримання своєї частини
    std::vector<int> sub_vector(ELEMENTS_PER_PROC);

    if (rank == ROOT_PROCESS) {
        long_vector.resize(vector_size);
        // Ініціалізуємо довгий вектор (наприклад, 1, 2, 3, ...)
        for (int i = 0; i < vector_size; ++i) {
            long_vector[i] = i + 1;
        }
        printf("--- Завдання 1: Розподілена Сума ---\n");
        printf("Root: Загальний розмір вектора: %d (%d елементів на %d процес(ів))\n",
            vector_size, ELEMENTS_PER_PROC, numtasks);
    }

    // 1. Розподілити вектор з кореневого процесу на всі інші
    // [cite: 156, 159]
    MPI_Scatter(
        long_vector.data(),    // буфер відправлення (значущий тільки для root) [cite: 161, 162]
        ELEMENTS_PER_PROC,     // кількість елементів, що надсилаються КОЖНОМУ процесу [cite: 163, 165]
        MPI_INT,               // тип даних відправлення
        sub_vector.data(),     // буфер прийому [cite: 163]
        ELEMENTS_PER_PROC,     // кількість елементів, що приймаються [cite: 163, 164]
        MPI_INT,               // тип даних прийому
        ROOT_PROCESS,          // ранг кореневого процесу-відправника [cite: 163]
        MPI_COMM_WORLD         // комунікатор
    );

    // 2. Обчислити часткову суму
    // Використовуємо std::accumulate для сумування елементів у sub_vector
    int partial_sum = std::accumulate(sub_vector.begin(), sub_vector.end(), 0);

    printf("Процес %d: Отримав %d елементів. Моя часткова сума: %d\n",
        rank, (int)sub_vector.size(), partial_sum);

    // 3. Зібрати всі часткові суми і додати їх у кореневому вузлі
    // [cite: 23, 263, 265]
    int total_sum = 0;
    MPI_Reduce(
        &partial_sum,      // буфер відправлення (локальна сума) [cite: 266]
        &total_sum,        // буфер прийому (значущий тільки для root) [cite: 266]
        1,                 // кількість елементів (одна сума від кожного) [cite: 266]
        MPI_INT,           // тип даних
        MPI_SUM,           // операція редукції (сума) [cite: 266, 287]
        ROOT_PROCESS,      // ранг кореневого процесу-одержувача [cite: 266]
        MPI_COMM_WORLD     // комунікатор
    );

    // Кореневий процес друкує остаточний результат
    if (rank == ROOT_PROCESS) {
        printf("--------------------------------------\n");
        printf("Root: Загальна сума (обчислена через MPI_Reduce): %d\n", total_sum);
        printf("--------------------------------------\n");
    }
}

/**
 * @brief Вирішує Завдання 2:
 * 1. Кожен з 'n' процесів має одне число (x_i).
 * 2. Використовує MPI_Scan для обчислення префіксної суми[cite: 305, 306, 334].
 * 3. Кожен процес 'i' отримує суму значень процесів 0...i.
 */
void solve_problem_2(int rank, int numtasks) {
    if (rank == 0) {
        printf("\n--- Завдання 2: Префіксна Сума (MPI_Scan) ---\n");
    }

    // Припустимо, n чисел на n процесах - це по одному числу на процес.
    // Для прикладу, процес 'rank' буде мати значення 'rank + 1'.
    int local_value = rank + 1;
    int prefix_sum = 0;

    // 2. Обчислити префіксну суму за допомогою MPI_Scan
    // [cite: 305, 307]
    MPI_Scan(
        &local_value,      // буфер відправлення (локальне значення) [cite: 308]
        &prefix_sum,       // буфер прийому (результат префіксної суми) [cite: 309]
        1,                 // кількість елементів [cite: 309]
        MPI_INT,           // тип даних [cite: 310]
        MPI_SUM,           // операція редукції (сума) [cite: 311, 287]
        MPI_COMM_WORLD     // комунікатор [cite: 312]
    );

    // Використовуємо бар'єри для синхронізації виводу,
    // щоб повідомлення друкувалися по порядку рангів.
    for (int i = 0; i < numtasks; ++i) {
        MPI_Barrier(MPI_COMM_WORLD); // 
        if (rank == i) {
            // y_i = x_0 + x_1 + ... + x_i [cite: 336, 337]
            printf("Процес %d: Локальне значення (x_%d) = %d. Префіксна сума (y_%d) = %d\n",
                rank, rank, local_value, rank, prefix_sum);
        }
    }
    if (rank == 0) {
        printf("--------------------------------------\n");
    }
}

int main(int argc, char** argv) {
    // Ініціалізація MPI [cite: 95]
    MPI_Init(&argc, &argv);

    int rank, numtasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Отримати ранг поточного процесу [cite: 97]
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks); // Отримати загальну кількість процесів [cite: 96]

    // --- Вирішення Завдання 1 ---
    solve_problem_1(rank, numtasks);

    // Синхронізувати всі процеси перед початком наступного завдання
    // [cite: 14, 43, 45]
    MPI_Barrier(MPI_COMM_WORLD);

    // --- Вирішення Завдання 2 ---
    solve_problem_2(rank, numtasks);

    // Завершення роботи з MPI [cite: 103]
    MPI_Finalize();
    return 0;
}