#include "renderer.h"
#include <cstdlib>
#include <cmath>

// Generatore di cerchi casuali
Circle circleGenerator() {
    Circle c;
    c.center.x = (float)(rand()) / ((float)(RAND_MAX) / (WIDTH));
    c.center.y = (float)(rand()) / ((float)(RAND_MAX) / (HEIGHT));
    c.center.z = (rand() % 25);
    c.radius   = (float)(rand()) / ((float)(RAND_MAX) / (70));
    c.red      = rand() % 255;
    c.green    = rand() % 255;
    c.blue     = rand() % 255;
    return c;
}

/* Distanza tra due punti in 2D */
float distance(float x1, float x2, float y1, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return sqrt(dx*dx + dy*dy);
}

/* Restituisce i pixel appartenenti a un cerchio */
std::vector<int> find_pixels_in_circle(const Circle &c) {
    std::vector<int> x;
    std::vector<int> y;

    int top_left_x    = (int)(c.center.x - c.radius);
    int top_left_y    = (int)(c.center.y - c.radius);
    int top_right_x   = (int)(c.center.x + c.radius);
    int bottom_left_y = (int)(c.center.y + c.radius);

    int edge = int(-c.radius * (1 - sqrt(2)));

    for(int i = top_left_y; i <= bottom_left_y; i++) {
        for(int j = top_left_x; j <= top_right_x; j++) {
            if(i >= 0 && j >= 0 && i < HEIGHT && j < WIDTH) {
                if((i > top_left_y + edge && i < bottom_left_y - edge) &&
                   (j > top_left_x + edge && j < top_right_x - edge)) {
                    x.push_back(j);
                    y.push_back(i);
                } else {
                    float dist = distance(c.center.x, j + 0.5f,
                                          c.center.y, i + 0.5f);
                    if(dist <= c.radius) {
                        x.push_back(j);
                        y.push_back(i);
                    }
                }
            }
        }
    }

    std::vector<int> coordinates;
    coordinates.insert(coordinates.end(), x.begin(), x.end());
    coordinates.insert(coordinates.end(), y.begin(), y.end());
    return coordinates;
}

/* Crea un vettore di valori alpha */
std::vector<double> create_alpha_value(std::vector<int> z) {
    std::vector<double> alphas(z.size());
    double tot = 1;

    int last = 0;
    bool stop = false;

    for(int i = 0; i < (int)z.size() - 1 && !stop; i++) {
        if(z[i] == z[i+1]) {
            last++;
        } else stop = true;
    }

    for(int i = (int)z.size() - 1; i >= last; i--) {
        int same = 0;
        if(i != last) {
            stop = false;
            for(int s = i; s > 0 && !stop; s--) {
                if(z[s] == z[s-1]) same++;
                else stop = true;
            }
            for(int d = 0; d <= same; d++) {
                alphas[i-d] = (ALPHA * tot)/(same+1);
            }
            tot -= ALPHA * tot;
            i -= same;
        } else {
            for(int l = 0; l <= last; l++) {
                alphas[l] = tot/(last+1);
            }
        }
        stop = false;
    }
    return alphas;
}

/* Inserimento ordinato in un vettore */
int ordered_insertion(std::vector<int> z, int z_coo) {
    if(z.empty()) return 0;

    int low = 0;
    int high = (int)z.size() - 1;

    while(low <= high) {
        int middle = low + (high - low) / 2;
        if(z_coo < z[middle]) high = middle - 1;
        else low = middle + 1;
    }
    return low;
}
