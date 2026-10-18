// ============================================================================
// ПРОСТАЯ НЕЙРОННАЯ СЕТЬ НА C++ (с подробными объяснениями)
// ============================================================================
//
// Что мы построим:
//   Полносвязная (dense) нейронная сеть из 3 слоёв:
//     вход (2 нейрона) -> скрытый слой (4 нейрона) -> выход (1 нейрон)
//
// Задача для обучения: функция XOR (исключающее ИЛИ).
//   Это классическая "игрушечная" задача, которую нельзя решить одним
//   линейным классификатором — как раз нужен скрытый слой.
//
//   Вход (a, b) | Ожидаемый вывод
//   ----------|------------------
//      0, 0    |       0
//      0, 1    |       1
//      1, 0    |       1
//      1, 1    |       0
//
// Как это работает (кратко):
//   1. ПРЯМОЕ РАСПРОСТРАНЕНИЕ (forward pass):
//      Каждый нейрон вычисляет взвешенную сумму входов + смещение (bias),
//      затем пропускает результат через функцию активации Sigmoid:
//          z = W * x + b
//          a = sigmoid(z) = 1 / (1 + e^(-z))
//      Sigmoid "сжимает" любое число в диапазон (0..1), что удобно,
//      когда хотим интерпретировать вывод как вероятность.
//
//   2. ОШИБКА (loss function):
//      Считаем среднеквадратичную ошибку (MSE):
//          E = sum over все примеры: (ожидаемое - полученное)^2
//      Чем меньше E, тем ближе сеть к правильному ответу.
//
//   3. ОБРАТНОЕ РАСПРОСТРАНЕНИЕ (backpropagation):
//      Ключевая идея обучения. Используем цепочку правил дифференцирования
//      из матанализа, чтобы понять, КАК каждый вес влияет на ошибку.
//      Для каждого веса считаем градиент dE/dW — «куда и насколько нужно
//      подвинуть вес, чтобы ошибка уменьшилась».
//
//   4. ГРАДИЕНТНЫЙ СПУСК (gradient descent):
//      Обновляем каждый вес в сторону, противоположную градиенту:
//          W := W - learning_rate * dE/dW
//      Повторяем шаги 1-4 много раз (эпохи), и сеть постепенно обучается.
//
// Компиляция и запуск:
//     g++ -std=c++17 -O2 -o nn neural_network.cpp && ./nn
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>    // Needed for exp() in the sigmoid function
#include <random>   // For random weight initialization

using namespace std;

// ----------------------------------------------------------------------------
// Вспомогательные функции активации и их производные
// ----------------------------------------------------------------------------

// Sigmoid: сжимает число в диапазон (0, 1).
//   sigmoid(0) = 0.5, sigmoid(+большое) ~ 1, sigmoid(-большое) ~ 0
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Производная sigmoid. Её хитрость: если y = sigmoid(x), то
//   y' = y * (1 - y)
// Поэтому после прямого распространения у нас уже есть y, и производная
// считается бесплатно — не нужно хранить x.
double sigmoid_derivative(double y) {
    return y * (1.0 - y);
}

// ----------------------------------------------------------------------------
// Класс Layer — один слой нейронной сети
// ----------------------------------------------------------------------------
// Слой хранит:
//   weights_         — матрица весов [выходов][входов]
//   biases_          — вектор смещений [выходов]
//   input_           — то, что подали на вход слоя (нужно для backprop)
//   output_          — результат работы слоя (активации нейронов)
//   grad_weights_, grad_biases_ — накопленные градиенты ошибок по весам/смещениям
class Layer {
public:
    vector<vector<double>> weights_;   // W[о][и] — вес от нейрона `и` входа к нейрону `о` выхода
    vector<double> biases_;            // b[о]    — смещение нейрона выхода `о`
    vector<double> input_;             // копия входа слоя (для обратного прохода)
    vector<double> output_;            // активации нейронов слоя
    vector<vector<double>> grad_weights_; // dE/dW — градиенты ошибки по весам
    vector<double> grad_biases_;            // dE/db — градиенты ошибки по смещениям

    int n_inputs_;
    int n_outputs_;

    // Конструктор: создаёт слой и инициализирует веса небольшими
    // случайными значениями. Важно: веса нельзя обнулять — иначе все
    // нейроны слоя будут вычислять одно и то же («симметрия» сломает обучение).
    Layer(int n_inputs, int n_outputs, mt19937& rng)
        : n_inputs_(n_inputs), n_outputs_(n_outputs)
    {
        uniform_real_distribution<double> dist(-1.0, 1.0);

        weights_.assign(n_outputs, vector<double>(n_inputs));
        biases_.assign(n_outputs, 0.0);
        grad_weights_.assign(n_outputs, vector<double>(n_inputs, 0.0));
        grad_biases_.assign(n_outputs, 0.0);

        for (int o = 0; o < n_outputs; ++o)
            for (int i = 0; i < n_inputs; ++i)
                weights_[o][i] = dist(rng);
    }

    // Прямое распространение: out = sigmoid(W * in + b)
    vector<double> forward(const vector<double>& in) {
        input_ = in;                    // запоминаем вход для backprop
        output_.assign(n_outputs_, 0.0);

        for (int o = 0; o < n_outputs_; ++o) {
            double z = biases_[o];      // начинаем со смещения
            for (int i = 0; i < n_inputs_; ++i)
                z += weights_[o][i] * in[i];  // сумма произведений вес*вход
            output_[o] = sigmoid(z);    // применяем функцию активации
        }
        return output_;
    }

    // Обратное распространение ПОСЛЕДНЕГО (выходного) слоя.
    // expected — желаемый ответ. Ошибка MSE: E = (expected - output)^2
    //   dE/da = -2 * (expected - output)
    //   delta = dE/dz = dE/da * sigmoid'(a)   — «ошибка», приписанная слою
    void backward_output(const vector<double>& expected) {
        for (int o = 0; o < n_outputs_; ++o) {
            double dE_da = -2.0 * (expected[o] - output_[o]);
            double delta = dE_da * sigmoid_derivative(output_[o]);

            // Градиент по весам: dE/dW[о][и] = delta * вход[и]
            // (чем сильнее вход «виноват» в ошибке, тем больше меняем его вес)
            for (int i = 0; i < n_inputs_; ++i)
                grad_weights_[o][i] = delta * input_[i];

            // Градиент по смещению: dE/db[о] = delta
            grad_biases_[o] = delta;
        }
    }

    // Обратное распространение СКРЫТОГО слоя.
    // Сюда приходит delta предыдущего (более глубокого) слоя, «размазанная»
    // по нашим выходам через цепное правило:
    //   dE/da[о] = sum over w: delta_w * W_hidden_to_w[о]
    void backward_hidden(const vector<double>& next_deltas, const vector<vector<double>>& next_weights) {
        for (int o = 0; o < n_outputs_; ++o) {
            // Суммарный вклад всех последующих нейронов в ошибку нашего нейрона o
            double dE_da = 0.0;
            for (size_t w = 0; w < next_deltas.size(); ++w)
                dE_da += next_deltas[w] * next_weights[w][o];

            double delta = dE_da * sigmoid_derivative(output_[o]);

            for (int i = 0; i < n_inputs_; ++i)
                grad_weights_[o][i] = delta * input_[i];
            grad_biases_[o] = delta;
        }
    }

    // Применение градиентного спуска: сдвигаем каждый параметр
    // против направления градиента (то есть в сторону уменьшения ошибки).
    void update(double lr) {
        for (int o = 0; o < n_outputs_; ++o) {
            for (int i = 0; i < n_inputs_; ++i)
                weights_[o][i] -= lr * grad_weights_[o][i];
            biases_[o] -= lr * grad_biases_[o];
        }
    }
};

// ----------------------------------------------------------------------------
// Класс NeuralNetwork — собирает слои вместе
// ----------------------------------------------------------------------------
class NeuralNetwork {
public:
    vector<Layer> layers_;

    NeuralNetwork(vector<int> sizes, unsigned seed = 42) {
        mt19937 rng(seed);
        for (size_t i = 0; i + 1 < sizes.size(); ++i)
            layers_.emplace_back(sizes[i], sizes[i + 1], rng);
    }

    // Прогон данных через всю сеть слева направо
    vector<double> forward(const vector<double>& x) {
        vector<double> cur = x;
        for (auto& layer : layers_)
            cur = layer.forward(cur);
        return cur;
    }

    // Обучение одного примера: forward -> backward -> update
    void train_step(const vector<double>& x, const vector<double>& y, double lr) {
        forward(x);  // теперь во всех слоях лежат input_/output_

        // 1) Backward выходного слоя. После этой функции в grad_biases_
        //    выходного слоя лежат его delta-значения (dE/dz).
        layers_.back().backward_output(y);

        // 2) Backward скрытых слоёв — идём справа налево.
        //    Дельты «следующего» слоя берём из его grad_biases_, потому что
        //    по определению grad_biases_[o] == delta_o этого нейрона.
        for (int l = static_cast<int>(layers_.size()) - 2; l >= 0; --l) {
            const auto& next = layers_[l + 1];
            vector<double> next_deltas(next.grad_biases_);  // это и есть delta'ы
            layers_[l].backward_hidden(next_deltas, next.weights_);
        }

        // 3) Обновление всех весов (градиентный спуск)
        for (auto& layer : layers_)
            layer.update(lr);
    }

    // Полный цикл обучения: много эпох по всем примерам
    void fit(const vector<vector<double>>& X,
             const vector<vector<double>>& Y,
             int epochs, double lr) {
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            double total_error = 0.0;

            for (size_t s = 0; s < X.size(); ++s) {
                train_step(X[s], Y[s], lr);

                // Считаем ошибку этого примера (MSE)
                auto pred = forward(X[s]);
                for (size_t k = 0; k < pred.size(); ++k)
                    total_error += (Y[s][k] - pred[k]) * (Y[s][k] - pred[k]);
            }

            if (epoch % 1000 == 0 || epoch == 1)
                cout << "Эпоха " << epoch << ", суммарная ошибка: "
                     << total_error << "\n";
        }
    }
};

// ----------------------------------------------------------------------------
// main — собираем всё вместе и решаем XOR
// ----------------------------------------------------------------------------
int main() {
    // Данные: 4 комбинации входов и правильные ответы для XOR
    vector<vector<double>> X = {{0,0}, {0,1}, {1,0}, {1,1}};
    vector<vector<double>> Y = {{0},   {1},   {1},   {0}};

    // Архитектура: 2 входа -> 4 скрытых -> 1 выход
    NeuralNetwork nn({2, 4, 1});

    cout << "=== Обучение сети задаче XOR ===\n";
    nn.fit(X, Y, /*epochs=*/10000, /*lr=*/0.5);

    cout << "\n=== Результаты после обучения ===\n";
    for (size_t i = 0; i < X.size(); ++i) {
        auto pred = nn.forward(X[i]);
        cout << "Вход (" << X[i][0] << ", " << X[i][1] << ") -> "
             << "ожидание " << Y[i][0]
             << ", сеть выдала " << pred[0]
             << " (округлённо: " << (pred[0] > 0.5 ? 1 : 0) << ")\n";
    }
    return 0;
}
