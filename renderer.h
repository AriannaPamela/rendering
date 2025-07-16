#ifndef RENDER_H
#define RENDER_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <omp.h>

const int WIDTH = 1080;
const int HEIGHT = 720;
const int SECTIONS = 16; //image divided in parts to compute rendering
const int SECTION_WIDTH = WIDTH/SECTIONS;
const int SECTION_HEIGHT = HEIGHT/SECTIONS;
const double ALPHA = 0.70; //alpha value for transparency
const int NUM_CIRCLES = 10000;


struct Point {
    float x; //coordiante along x axe
    float y; //coordiante along y axe
    int z ; //coordiante along z axe
};

struct Circle {
    Point center;
    float radius;
    int red;
    int green;
    int blue;
};

Circle circleGenerator();

/*Distance between two points in 2d space.*/
float distance(float x1, float x2, float y1, float y2);

/*Function that return a vector containing the coordinates of the pixels belonging to a circle c*/
std::vector<int> find_pixels_in_circle(const Circle &c);

/*Create an array with alpha layer value.*/
std::vector<double> create_alpha_value(std::vector<int> z);

/*Insert a value in a sorted vector*/
int ordered_insertion(std::vector<int> z, int z_coo);

#endif