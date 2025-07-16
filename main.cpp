#include <iostream>
#include <random>
#include <time.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <fstream>
#include <numeric>
#include <algorithm> // Per std::min e std::max

#include "renderer.h" // Includi il tuo header per le funzioni e strutture definite lì

int main() {
    srand(49);
    int thread = 16;

    omp_set_num_threads(thread);

    std::cout << "Using # " << omp_get_max_threads() << " threads." << std::endl;
    std::cout << "Using # " << omp_get_num_procs() << " core." << std::endl;

    cv::Mat sequential_image = cv::Mat(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat parallel_image = cv::Mat(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));

    /*Generate all circles*/
    Circle all_circles[NUM_CIRCLES];
    for (int i = 0; i < NUM_CIRCLES; i++) {
        all_circles[i] = circleGenerator();
    }

    sequential_image = cv::Mat(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
    parallel_image = cv::Mat(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));

    double start_total_seq = omp_get_wtime();

    std::vector<Circle> circles_in_sections_seq[SECTIONS * SECTIONS];
    // array of vectors with circles divided in image sections

    double start_mapping_sections_seq = omp_get_wtime();

    for (int i = 0; i < NUM_CIRCLES; i++) {
        Circle c = all_circles[i];

        //Each circle must be added in the section he laies on, both the section where is his center, also the adiacent sections he can reach
        int min_col = std::max(0, static_cast<int>(((c.center.x - c.radius) / SECTION_WIDTH)));
        //the min index of column section occupied by this circle.
        int max_col = std::min(SECTIONS - 1, static_cast<int>(((c.center.x + c.radius) / SECTION_WIDTH)));
        //the max index of column section occupied by this circle
        int min_row = std::max(0, static_cast<int>(((c.center.y - c.radius) / SECTION_HEIGHT)));
        //the min index of row section occupied by this circle
        int max_row = std::min(SECTIONS - 1, static_cast<int>(((c.center.y + c.radius) / SECTION_HEIGHT)));
        //the max index of row section occupied by this circle


        // Add circle to his sections
        for (int row = min_row; row <= max_row; row++) {
            for (int col = min_col; col <= max_col; col++) {
                int section_index = row * SECTIONS + col;
                circles_in_sections_seq[section_index].push_back(c);
            }
        }
    }
    double end_mapping_sections_seq = omp_get_wtime();

    double start_pixel_rendering;
    double stop_pixel_rendering;

    double seq_map_pix = 0.0;
    double seq_render_pix = 0.0;

    for (int h = 0; h < SECTIONS; h++) {
        for (int w = 0; w < SECTIONS; w++) {
            double t1 = omp_get_wtime();
            /*Consider real dimension of this section to avoid black column/rows at the end of the image*/
            int current_section_width = (w == SECTIONS - 1) ? (WIDTH - w * SECTION_WIDTH) : SECTION_WIDTH;
            int current_section_height = (h == SECTIONS - 1) ? (HEIGHT - h * SECTION_HEIGHT) : SECTION_HEIGHT;

            // Matrix to store circles lying in each pixels. Dimensions equals to image section considered. In each cell there is an array of 4 elements, all r,g,b,z elements laying in that pixel.
            std::vector<int> pixel_circle_data[current_section_height][current_section_width][4];

            /*Circle mapping in pixels*/
            for (int n = 0; n < circles_in_sections_seq[h * SECTIONS + w].size(); n++) {
                Circle *circle = &circles_in_sections_seq[h * SECTIONS + w][n];

                std::vector<int> coordinates = find_pixels_in_circle(*circle);
                //Find pixels belonging to this circle
                int n_pixels = coordinates.size() / 2; //number of pixels in the circle

                /*Add the circle to pixels he laies on*/
                for (int i = 0; i < n_pixels; i++) {
                    /*Convert to general image coordinates*/
                    int x_coo = coordinates[i] - w * SECTION_WIDTH;
                    int y_coo = coordinates[i + n_pixels] - h * SECTION_HEIGHT;

                    /*Add circle according to z order*/
                    if (x_coo >= 0 && y_coo >= 0 && x_coo < current_section_width && y_coo <
                        current_section_height) {
                        int index = ordered_insertion(pixel_circle_data[y_coo][x_coo][0], circle->center.z);

                        pixel_circle_data[y_coo][x_coo][0].insert(
                            pixel_circle_data[y_coo][x_coo][0].begin() + index, circle->center.z);
                        pixel_circle_data[y_coo][x_coo][1].insert(
                            pixel_circle_data[y_coo][x_coo][1].begin() + index, circle->red);
                        pixel_circle_data[y_coo][x_coo][2].insert(
                            pixel_circle_data[y_coo][x_coo][2].begin() + index, circle->green);
                        pixel_circle_data[y_coo][x_coo][3].insert(
                            pixel_circle_data[y_coo][x_coo][3].begin() + index, circle->blue);
                    }
                }
            }
            for (int i = 0; i < current_section_height; ++i) {
                for (int j = 0; j < current_section_width; ++j) {
                    pixel_circle_data[i][j][0].shrink_to_fit(); //Red channel
                    pixel_circle_data[i][j][1].shrink_to_fit(); //Green channel
                    pixel_circle_data[i][j][2].shrink_to_fit(); //Blue channel
                    pixel_circle_data[i][j][3].shrink_to_fit(); //Z channel
                }
            }

            double t2 = omp_get_wtime();
            seq_map_pix += t2 - t1;

            /*Pixel rendering*/
            double t3 = omp_get_wtime();
            start_pixel_rendering = omp_get_wtime();
            for (int i = 0; i < current_section_height; i++) {
                for (int j = 0; j < current_section_width; j++) {
                    if (!pixel_circle_data[i][j][0].empty()) {
                        std::vector<double> alphas = create_alpha_value(pixel_circle_data[i][j][0]);
                        // alpha values for this pixel
                        alphas.shrink_to_fit();

                        /*Convert i,j coordinates to general image coordinates*/
                        int x = j + w * SECTION_WIDTH;
                        int y = i + h * SECTION_HEIGHT;

                        for (int z = 0; z < pixel_circle_data[i][j][0].size(); z++) {
                            /*OpenCV BGR images */
                            int previous_red = sequential_image.at<cv::Vec3b>(y, x)[2];
                            int previous_green = sequential_image.at<cv::Vec3b>(y, x)[1];
                            int previous_blue = sequential_image.at<cv::Vec3b>(y, x)[0];

                            sequential_image.at<cv::Vec3b>(y, x) = cv::Vec3b(
                                previous_blue + pixel_circle_data[i][j][3][z] * alphas[z],
                                previous_green + pixel_circle_data[i][j][2][z] * alphas[z],
                                previous_red + pixel_circle_data[i][j][1][z] * alphas[z]
                            );
                        }
                    }
                }
            }

            double t4 = omp_get_wtime();
            seq_render_pix += (t4 - t3);
        }
    }
    double end_total_seq = omp_get_wtime();
    cv::imwrite("sequential_circle_image.png", sequential_image);


    /* -------------PARALLEL VERSION----------------*/
    double start_total_par = omp_get_wtime();

    std::vector<Circle> circles_in_sections_par[SECTIONS * SECTIONS];
    //Array of circle organized in image sections
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<Circle> > local_circles[num_threads];
    //Thread-local storage of circle in image sections

    for (int i = 0; i < num_threads; ++i) {
        local_circles[i].resize(SECTIONS * SECTIONS);
    }

    double start_mapping_sections_par = omp_get_wtime();
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto &my_local = local_circles[tid]; //local copy of this thread

#pragma omp for
        for (int i = 0; i < NUM_CIRCLES; ++i) {
            Circle c = all_circles[i];

            int min_col = std::max(0, static_cast<int>((c.center.x - c.radius) / SECTION_WIDTH));
            int max_col = std::min(SECTIONS - 1, static_cast<int>((c.center.x + c.radius) / SECTION_WIDTH));

            int min_row = std::max(0, static_cast<int>((c.center.y - c.radius) / SECTION_HEIGHT));
            int max_row = std::min(SECTIONS - 1, static_cast<int>((c.center.y + c.radius) / SECTION_HEIGHT));

            for (int row = min_row; row <= max_row; ++row) {
                for (int col = min_col; col <= max_col; ++col) {
                    int section_index = row * SECTIONS + col;
                    my_local[section_index].push_back(c);
                }
            }
        }

        // merge all local sections
#pragma omp single
        {
            for (int s = 0; s < SECTIONS * SECTIONS; ++s) {
                for (int t = 0; t < num_threads; ++t) {
                    circles_in_sections_par[s].insert(circles_in_sections_par[s].end(),
                                                      local_circles[t][s].begin(), local_circles[t][s].end());
                }
            }
        }
    }
    double end_mapping_sections_par = omp_get_wtime();

    std::vector<double> map_pix_per_thread(thread, 0);
    std::vector<double> render_pix_per_thread(thread, 0);

    double start_section_processing = omp_get_wtime();
#pragma omp parallel for collapse(2)
    for (int h = 0; h < SECTIONS; h++) {
        for (int w = 0; w < SECTIONS; w++) {
            int tid = omp_get_thread_num();
            double start_mapping_circle_pixel = omp_get_wtime();
            /*Consider real dimension of this section to avoid black column/rows at the end of the image*/
            int current_section_width = (w == SECTIONS - 1) ? (WIDTH - w * SECTION_WIDTH) : SECTION_WIDTH;
            int current_section_height = (h == SECTIONS - 1) ? (HEIGHT - h * SECTION_HEIGHT) : SECTION_HEIGHT;

            std::vector<int> pixel_circle_data[current_section_height][current_section_width][4];
            // Matrix to store circles lying in each pixels. Dimensions equals to image section considered. In each cell there is an array of 4 elements, all r,g,b,z elements laying in that pixel.

            /*Computation of pixel coordinates belonging to each circle*/
            auto circle_vector = circles_in_sections_par[h * SECTIONS + w];
            for (int n = 0; n < circle_vector.size(); n++) {
                Circle circle = circle_vector[n];
                std::vector<int> coordinates = find_pixels_in_circle(circle);
                //Find pixels belonging to this circle
                int n_pixels = coordinates.size() / 2; //number of pixels in the circle
                /*Add the circle to pixels he lies on*/
                for (int i = 0; i < n_pixels; i++) {
                    /*Convert to general image coordinates*/
                    int x_coo = coordinates[i] - w * SECTION_WIDTH;
                    int y_coo = coordinates[i + n_pixels] - h * SECTION_HEIGHT;

                    /*Add circle according to z order*/
                    if (x_coo >= 0 && y_coo >= 0 && x_coo < current_section_width && y_coo <
                        current_section_height) {
                        int index = ordered_insertion(pixel_circle_data[y_coo][x_coo][0], circle.center.z);

                        pixel_circle_data[y_coo][x_coo][0].insert(
                            pixel_circle_data[y_coo][x_coo][0].begin() + index, circle.center.z);
                        pixel_circle_data[y_coo][x_coo][1].insert(
                            pixel_circle_data[y_coo][x_coo][1].begin() + index, circle.red);
                        pixel_circle_data[y_coo][x_coo][2].insert(
                            pixel_circle_data[y_coo][x_coo][2].begin() + index, circle.green);
                        pixel_circle_data[y_coo][x_coo][3].insert(
                            pixel_circle_data[y_coo][x_coo][3].begin() + index, circle.blue);
                    }
                }
            }

            for (int i = 0; i < current_section_height; ++i) {
                for (int j = 0; j < current_section_width; ++j) {
                    pixel_circle_data[i][j][0].shrink_to_fit(); //Red channel
                    pixel_circle_data[i][j][1].shrink_to_fit(); //Green channel
                    pixel_circle_data[i][j][2].shrink_to_fit(); //Blue Channel
                    pixel_circle_data[i][j][3].shrink_to_fit(); //Z Channel
                }
            }

            double stop_mapping_circle_pixel = omp_get_wtime();
            map_pix_per_thread[tid] += stop_mapping_circle_pixel - start_mapping_circle_pixel;

            double start_rendering_pixel = omp_get_wtime();

            /*Parallelize pixel color computation*/
            for (int i = 0; i < current_section_height; i++) {
                for (int j = 0; j < current_section_width; j++) {
                    /*Count elements with same z coordinates in this pixel*/
                    if (!pixel_circle_data[i][j][0].empty()) {
                        std::vector<double> alphas = create_alpha_value(pixel_circle_data[i][j][0]);
                        // alpha value for this pixel
                        alphas.shrink_to_fit();

                        /*Convert i,j coordinates to general image coordinates*/
                        int x = j + w * SECTION_WIDTH;
                        int y = i + h * SECTION_HEIGHT;

                        for (int z = 0; z < pixel_circle_data[i][j][0].size(); z++) {
                            /*Store previous pixel color, needed to add new circle contribute to the pixel*/
                            int previous_red = parallel_image.at<cv::Vec3b>(y, x)[2];
                            int previous_green = parallel_image.at<cv::Vec3b>(y, x)[1];
                            int previous_blue = parallel_image.at<cv::Vec3b>(y, x)[0];
                            double alpha = alphas[z];

                            parallel_image.at<cv::Vec3b>(y, x) = cv::Vec3b(
                                previous_blue + pixel_circle_data[i][j][3][z] * alpha,
                                previous_green + pixel_circle_data[i][j][2][z] * alpha,
                                previous_red + pixel_circle_data[i][j][1][z] * alpha
                            );
                        }
                    }
                }
            }
            double stop_rendering_pixel = omp_get_wtime();
            render_pix_per_thread[tid] += stop_rendering_pixel - start_rendering_pixel;
        }
    }


    double stop_section_processing = omp_get_wtime();

    auto max_map = map_pix_per_thread[0];
    for (int i = 0; i < thread; i++) {
        if (map_pix_per_thread[i] > max_map)
            max_map = map_pix_per_thread[i];
    }

    auto max_ren = render_pix_per_thread[0];
    for (int i = 0; i < thread; i++) {
        if (render_pix_per_thread[i] > max_ren)
            max_ren = render_pix_per_thread[i];
    }

    double end_total_par = omp_get_wtime();
    cv::imwrite("parallel_circle_image.png", parallel_image);


    // Check correctness
    cv::Mat diff;
    absdiff(sequential_image, parallel_image, diff);

    if (countNonZero(diff.reshape(1)) == 0) {
        std::cout << "Correct rendering" << std::endl;
    } else {
        std::cout << "Wrong rendering" << std::endl;
    }

    return 0;
}
