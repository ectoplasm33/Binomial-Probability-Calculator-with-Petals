#include <iostream>
#include <SDL3/SDL.h>
#include <SDL3_ttf/SDL_ttf.h>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <SDL3_image/SDL_image.h>
#include <boost/math/special_functions/beta.hpp>
#include <thread>
#include <barrier>

int win_x = 1500; int win_y = std::round(win_x * 9.0f / 16.0f);
float hwin_x = win_x / 2.0f; float hwin_y = win_y / 2.0f;

double zoom = 1.0;
float cam_x = win_x / 2.0f;
float cam_y = win_y / 2.0f;

const float fps_limit = 65;
const float delay = 1000.0f / fps_limit;

double num_trials = 100;
double current_trials = num_trials;
double petals = num_trials;
double chance = 0.01;
double threshold = 1e-8;
int calculated_outcomes;
int last;
double expected;
uint64_t chunk;

std::vector<double> raw_probabilities;
std::vector<double> n_plus_prob;
std::vector<double> n_minus_prob;

std::vector<double> equation_raw_probs;
std::vector<double> equation_n_plus_probs;
std::vector<double> equation_n_minus_probs;

int decimal_precision = 4;
int equation_resolution = 5;
int equation_points;
bool input_active = false;
bool display_percents = false;
bool display_gridlines = true;
bool display_equation = false;
bool display_settings = false;
bool calculate_petals = false;

const int side_bar_width = 38;
const float half_bar = side_bar_width / 2.0f;

const float border = 30.0f;
const float double_border = border * 2.0f;
float label_spacing = 50.0f;

int mouse_x;
int mouse_y;

bool lmb = false;
bool drag = false;

unsigned int hc = std::thread::hardware_concurrency();

unsigned int thread_count = (hc > 1) ? (hc - 1) : 1;

std::barrier frame_barrier(thread_count + 1);

std::vector<std::thread> threads;

bool active = true;
std::string worker_task;

static SDL_Texture* create_circle_texture(SDL_Renderer* renderer, float r, Uint8 red, Uint8 green, Uint8 blue) {
	int diameter = std::ceil(r) * 2 + 2;

	SDL_Texture* circle = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_TARGET, diameter, diameter);
	SDL_SetTextureBlendMode(circle, SDL_BLENDMODE_BLEND);
	SDL_SetRenderTarget(renderer, circle);
	SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
	SDL_RenderClear(renderer);

	float dist;
	int half_diameter = r + 1.0f;
	float r2 = r * r;

	for (float dx = -half_diameter; dx < half_diameter + 1.0f; dx += 1.0f) {
		for (float dy = -half_diameter; dy < half_diameter + 1.0f; dy += 1.0f) {
			dist = dx * dx + dy * dy;
			if (dist < r2) {
				SDL_SetRenderDrawColor(renderer, red, green, blue, 255);
				SDL_RenderPoint(renderer, half_diameter + dx, half_diameter + dy);
			} else {
				dist = std::sqrt(dist);

				if (dist < r + 1.0f) {
					SDL_SetRenderDrawColor(renderer, red, green, blue, std::clamp((int)std::round((r + 1.0f - dist) * 255), 0, 255));
					SDL_RenderPoint(renderer, half_diameter + dx, half_diameter + dy);
				}
			}
		}
	}

	SDL_SetRenderTarget(renderer, nullptr);

	return circle;
}

static inline SDL_Texture* create_text_texture(SDL_Renderer* renderer, std::string text, TTF_Font* font, SDL_Color color = {0,0,0,255}) {
	int len = SDL_strnlen(text.c_str(), 500);

	SDL_Surface* surface = TTF_RenderText_Blended(font, text.c_str(), len, color);
	SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
	SDL_SetTextureBlendMode(texture, SDL_BLENDMODE_BLEND);

	SDL_DestroySurface(surface);

	return texture;
}

class button {
public:
	float x, y, w, h;
	SDL_FRect rect;
	SDL_Texture* icon;
	SDL_Texture* icon_select;

	button(float x_pos, float y_pos, float width, float height) {
		x = x_pos;
		y = y_pos;
		w = width;
		h = height;

		width = (int)std::floor(w) + 2;
		height = (int)std::floor(h) + 2;

		rect = {std::round(x) - 1,std::round(y) - 1,width,height};
	}
};

class toggle {
public:
	float x, y;
	bool* state;
	SDL_FRect rect;

	toggle(SDL_Renderer* renderer, float x_pos, float y_pos, float w, float h, bool* val) {
		x = x_pos;
		y = y_pos;

		state = val;

		rect = {x,y,w,h};
	}
};

class label {
public:
	float x, y;
	SDL_FRect rect;
	SDL_Texture* text;

	label(SDL_Renderer* renderer, float x_pos, float y_pos, std::string str, TTF_Font* font, SDL_Color color) {
		x = x_pos;
		y = y_pos;

		text = create_text_texture(renderer, str, font, color);
		float w, h;
		SDL_GetTextureSize(text, &w, &h);

		rect = {x, y, w, h};
	}
};

static inline double calc_avg(double num, double chance) {
	double t[6] = {0,0,0,0,0,0};

	double q = 1.0 - chance;

	int i;
	for (i = 5; i < num + 1; i++) {

		double success = t[(i - 5) % 6] * chance;

		double avg = (t[(i - 1) % 6] + t[(i - 2) % 6] + t[(i - 3) % 6] + t[(i - 4) % 6]) / 4.0;

		t[i % 6] = 1.0 + success + q * avg;
	}

	return t[(i - 1) % 6];
}

static inline double convert_trials_to_petals(double num, double chance) {
	double m = 0.4 - 0.4 * chance / (1 + chance);
	
	if (num < 1e5) {
		if (num == std::floor(num)) {
			return calc_avg(num, chance);
		}

		double avg = calc_avg(num, chance);

		return (num - std::floor(num)) * m + avg;
	}

	double b = calc_avg(1e5, chance) - 1e5 * m;

	return num * m + b;
}

static void execute_thread(unsigned int id) {
	uint64_t start, end;
	
	while (active) {
		frame_barrier.arrive_and_wait();

		if (!active) break;
		
		if (worker_task == "raw") {
			start = chunk * id;
			end = std::min((id + 1) * chunk, (uint64_t)std::ceil(expected));

			double log_factorial_trials = std::lgamma(current_trials + 1.0);

			double amount, prob, probability, index;
			for (uint64_t i = start; i < end; i++) {
				index = (double)i;

				amount = log_factorial_trials - std::lgamma(index + 1.0) - std::lgamma(current_trials - index + 1.0);
				prob = std::log(1.0 - chance) * (current_trials - index) + std::log(chance) * index;
				probability = std::exp(amount + prob);

				raw_probabilities[i] = probability;
			}
		} else if (worker_task == "n-") {
			start = chunk * id;
			end = std::min((id + 1) * chunk, (uint64_t)std::ceil(expected));

			double z = 1.0 - chance;

			for (uint64_t i = start; i < end; i++) {
				n_minus_prob[i] = boost::math::ibeta(current_trials - (double)i, (double)i + 1.0, z);
			}
		} else if (worker_task == "n+") {
			start = chunk * id;
			end = std::min((id + 1) * chunk, (uint64_t)std::ceil(expected));

			double z = 1.0 - chance;

			for (uint64_t i = start; i < end; i++) {
				n_plus_prob[i] = 1.0 - boost::math::ibeta(current_trials - (double)i + 1.0, (double)i, z);
			}
		} else if (worker_task == "eq raw") {
			start = chunk * id;
			end = std::min((id + 1) * chunk, (uint64_t)std::ceil(equation_points));

			double log_factorial_trials = std::lgamma(current_trials + 1.0);

			double amount, prob, probability, index;
			double inv_res = 1.0 / equation_resolution;
			for (size_t i = start; i < end; i++) {
				index = (double)(last + i * inv_res);

				amount = log_factorial_trials - std::lgamma(index + 1.0) - std::lgamma(current_trials - index + 1.0);
				prob = std::log(1.0 - chance) * (current_trials - index) + std::log(chance) * index;
				probability = std::exp(amount + prob);

				equation_raw_probs[i] = (probability);
			}

		} else if (worker_task == "eq n-") {
			start = chunk * id;
			end = std::min((id + 1) * chunk, (uint64_t)std::ceil(equation_points));

			double z = 1.0 - chance;

			double inv_res = 1.0 / equation_resolution;

			for (size_t i = start; i < end; i++) {
				double k = (double)i * inv_res + last;
				equation_n_minus_probs[i] = boost::math::ibeta(current_trials - k, k + 1.0, z);
			}

		} else if (worker_task == "eq n+") {
			start = chunk * id;
			end = std::min((id + 1) * chunk, (uint64_t)std::ceil(equation_points));

			double z = 1.0 - chance;

			double inv_res = 1.0 / equation_resolution;

			for (size_t i = start; i < end; i++) {
				double k = (double)i * inv_res + last;
				equation_n_plus_probs[i] = 1.0 - boost::math::ibeta(current_trials - k + 1.0, k, z);
			}
		}

		frame_barrier.arrive_and_wait();
	}
}

static inline void calculate_raw_probablilities() {
	expected = current_trials * chance;

	int exp = std::ceil(expected);

	chunk = std::ceil(exp / (float)thread_count);

	if (expected > 100000) {
		worker_task = "raw";

		raw_probabilities.resize(exp);

		frame_barrier.arrive_and_wait();

		frame_barrier.arrive_and_wait();

		double log_factorial_trials = std::lgamma(current_trials + 1.0);

		uint64_t possible_outcomes = ceil(current_trials) + 1;
		double amount, prob, probability, index;
		for (uint64_t i = exp; i < possible_outcomes; i++) {
			index = (double)i;

			amount = log_factorial_trials - std::lgamma(index + 1.0) - std::lgamma(current_trials - index + 1.0);
			prob = std::log(1.0 - chance) * (current_trials - index) + std::log(chance) * index;
			probability = std::exp(amount + prob);

			raw_probabilities.push_back(probability);

			if (probability < threshold) {
				break;
			}
		}

	} else {
		raw_probabilities.clear();
		raw_probabilities.reserve(exp);

		if (current_trials > 170.0) {
			double log_factorial_trials = std::lgamma(current_trials + 1.0);

			uint64_t possible_outcomes = ceil(current_trials) + 1;
			double amount, prob, probability, index;
			for (uint64_t i = 0; i < possible_outcomes; i++) {
				index = (double)i;

				amount = log_factorial_trials - std::lgamma(index + 1.0) - std::lgamma(current_trials - index + 1.0);
				prob = std::log(1.0 - chance) * (current_trials - index) + std::log(chance) * index;
				probability = std::exp(amount + prob);

				raw_probabilities.push_back(probability);

				if (probability < threshold && i > expected) {
					break;
				}
			}
		} else {
			double factorial_trials = std::tgamma(current_trials + 1.0);

			int possible_outcomes = (int)ceil(current_trials) + 1;
			double amount, prob, probability, index;
			for (size_t i = 0; i < possible_outcomes; i++) {
				index = (double)i;

				amount = factorial_trials / (std::tgamma(index + 1.0) * std::tgamma(current_trials - index + 1.0));
				prob = std::pow((1.0 - chance), (current_trials - index)) * std::pow(chance, index);

				probability = amount * prob;

				raw_probabilities.push_back(probability);

				if (probability < threshold && i > expected) {
					break;
				}
			}
		}
	}
}

static inline void calculate_n_minus_prob() {
	size_t size = raw_probabilities.size();
	n_minus_prob.resize(size);

	if (expected > 100000) {
		worker_task = "n-";

		frame_barrier.arrive_and_wait();

		frame_barrier.arrive_and_wait();

		double z = 1.0 - chance;

		for (size_t i = std::ceil(expected); i < size; i++) {
			n_minus_prob[i] = boost::math::ibeta(current_trials - (double)i, (double)i + 1.0, z);
		}

	} else {
		double sum = 0;

		for (size_t i = 0; i < size; i++) {
			sum += raw_probabilities[i];
			n_minus_prob[i] = sum;
		}
	}
}

static inline void calculate_n_plus_prob() {
	size_t size = raw_probabilities.size();
	n_plus_prob.resize(size);

	if (expected > 100000) {
		worker_task = "n+";

		frame_barrier.arrive_and_wait();

		frame_barrier.arrive_and_wait();

		double z = 1.0 - chance;

		for (size_t i = std::ceil(expected); i < size; i++) {
			n_plus_prob[i] = 1.0 - boost::math::ibeta(current_trials - (double)i + 1.0, (double)i, z);
		}

	} else {
		double sum = 1;

		for (int i = 0; i < size; i++) {
			n_plus_prob[i] = sum;
			sum -= raw_probabilities[i];
		}
	}	
}

static inline void approximate_raw_probs() {
	equation_points = (calculated_outcomes - 1) * equation_resolution + 1;

	if (equation_points > 100000) {
		equation_raw_probs.resize(equation_points);

		chunk = std::ceil(equation_points / (float)thread_count);

		worker_task = "eq raw";

		frame_barrier.arrive_and_wait();

		frame_barrier.arrive_and_wait();

	} else {
		equation_raw_probs.clear();
		equation_n_plus_probs.reserve(equation_points);

		if (current_trials > 170) {
			double log_factorial_trials = std::lgamma(current_trials + 1.0);

			double amount, prob, probability, index;
			double inv_res = 1.0 / equation_resolution;
			for (size_t i = 0; i < equation_points; i++) {
				index = (double)(last + i * inv_res);

				amount = log_factorial_trials - std::lgamma(index + 1.0) - std::lgamma(current_trials - index + 1.0);
				prob = std::log(1.0 - chance) * (current_trials - index) + std::log(chance) * index;
				probability = std::exp(amount + prob);

				equation_raw_probs.push_back(probability);
			}
		} else {
			double factorial_trials = std::tgamma(current_trials + 1.0);

			double amount, prob, probability, index;
			double inv_res = 1.0 / equation_resolution;
			for (size_t i = 0; i < equation_points; i++) {
				index = (double)(last + i * inv_res);

				amount = factorial_trials / (std::tgamma(index + 1.0) * std::tgamma(current_trials - index + 1.0));
				prob = std::pow((1.0 - chance), (current_trials - index)) * std::pow(chance, index);

				probability = amount * prob;

				equation_raw_probs.push_back(probability);
			}
		}
	}
}

static inline void approximate_n_minus_probs() {
	if (expected > 100000) {
		equation_n_minus_probs.resize(equation_points);

		chunk = std::ceil(equation_points / (float)thread_count);

		worker_task = "eq n-";

		frame_barrier.arrive_and_wait();

		frame_barrier.arrive_and_wait();
	} else {
		equation_n_minus_probs.clear();
		equation_n_minus_probs.reserve(equation_points);

		double z = 1.0 - chance;

		double inv_res = 1.0 / equation_resolution;

		for (size_t i = 0; i < equation_points; i++) {
			double k = (double)i * inv_res + last;
			equation_n_minus_probs.push_back(boost::math::ibeta(current_trials - k, k + 1.0, z));
		}
	}
}

static inline void approximate_n_plus_probs() {
	if (expected > 100000) {
		equation_n_plus_probs.resize(equation_points);

		chunk = std::ceil(equation_points / (float)thread_count);

		worker_task = "eq n+";

		frame_barrier.arrive_and_wait();

		frame_barrier.arrive_and_wait();
	} else {
		equation_n_plus_probs.clear();
		equation_n_plus_probs.reserve(equation_points);

		double z = 1.0 - chance;

		double inv_res = 1.0 / equation_resolution;

		for (size_t i = 0; i < equation_points; i++) {
			double k = (double)i * inv_res + last;
			equation_n_plus_probs.push_back(1.0 - boost::math::ibeta(current_trials - k + 1.0, k, z));
		}
	}
}

static inline bool valid_number(const std::string& s) {
	try {
		size_t pos;
		std::stod(s, &pos);
		return pos == s.length();
	} catch (...) {
		return false;
	}
}

// format value to a specified number of decimals 
// format atleast the specified amount for values <1
static inline std::string format_val(double value, int precision) {
	if (value < 1.0 && value > -1.0) {
		std::ostringstream out;
		out << std::fixed << std::setprecision(16) << value;

		std::string str = out.str();

		int i = 0;

		int offset = 2;

		if (str.front() == '-') {
			i++;
			offset = 3;
		}

		while ((str[i] == '0' || str[i] == '.') && i != str.size()) {
			i++;
		}

		std::ostringstream out2;

		out2 << std::fixed << std::setprecision(std::min(precision + i - offset, 16)) << value;

		str = out2.str();

		while (str.back() == '0') {
			str.pop_back();
		}

		if (str.back() == '.') {
			str.pop_back();
		}

		return str;
	} else {
		std::ostringstream out;
		out << std::fixed << std::setprecision(precision) << value;

		std::string str = out.str();

		while (str.back() == '0') {
			str.pop_back();
		}

		if (str.back() == '.') {
			str.pop_back();
		}

		return str;
	}
}

static std::vector<SDL_Texture*> create_vertical_labels(SDL_Renderer* renderer, TTF_Font* font, SDL_Color color) {
	std::vector<SDL_Texture*> labels;

	if (display_percents) {
		for (int i = 0; i < 11; i++) {
			labels.emplace_back(create_text_texture(renderer, format_val(i * 10, 1) + "%", font, color));
		}
	} else {
		for (int i = 0; i < 11; i++) {
			labels.emplace_back(create_text_texture(renderer, format_val(i * 0.1, 1), font, color));
		}
	}

	return labels;
}

static inline std::vector<SDL_Texture*> create_horizontal_labels(SDL_Renderer* renderer, int start, TTF_Font* font, SDL_Color color) {
	std::vector<SDL_Texture*> labels;

	size_t max_str_len = 0;

	for (int i = 0; i < calculated_outcomes + 1; i++) {
		std::string str = format_val(i + start, 1);

		max_str_len = std::max(max_str_len, str.size());

		labels.emplace_back(create_text_texture(renderer, str, font, color));
	}

	label_spacing = 50.0f + std::max((int)max_str_len - 7, 0) * 6.0f;

	return labels;
}

struct point {
	float x;
	double* y;
	int type;
};

struct relative_rect {
	float dx;
	float dy;
	float w;
	float h;
};

struct inspect_box {
	float origin_x;
	double* origin_y;
	float w;
	float h;
	point contents[12];
	SDL_Texture* textures[12];
	relative_rect text_rects[12];
	relative_rect indicator_rects[12];
	relative_rect render_rects[9];
	int size;
};

struct inspect_point {
	double dist;
	point pt;
};

struct mouse_inspect_box {
	float x;
	float y;
	float w;
	float h;
	point contents[10];
	SDL_Texture* textures[10];
	SDL_FRect text_rects[10];
	SDL_FRect indicator_rects[10];
	int size;
};

struct remove_pt {
	int box_idx;
	int pt_idx;
};

static inline void setup_mouse_inspect_box(SDL_Renderer* renderer, mouse_inspect_box* box, TTF_Font* font, SDL_Color color0, SDL_Color color1, SDL_Color color2) {
	float w, h;
	float sum_h = -3;
	SDL_Color color;
	std::string str;

	for (int i = 0; i < box->size; i++) {
		point* pt = &box->contents[i];

		if (pt->type == 0) {
			color = color0;
		} else if (pt->type == 1) {
			color = color1;
		} else {
			color = color2;
		}

		if (display_percents) {
			str = format_val(pt->x, 1) + ", " + format_val(*pt->y * 100, decimal_precision) + "%";
		} else {
			str = format_val(pt->x, 1) + ", " + format_val(*pt->y, decimal_precision);
		}

		box->textures[i] = create_text_texture(renderer, str, font, color);
		SDL_GetTextureSize(box->textures[i], &w, &h);
		sum_h += h + 3;

		box->text_rects[i] = SDL_FRect{box->x + 20, box->y - sum_h - 6, w, h};
		box->indicator_rects[i] = SDL_FRect{std::round(box->x + 7), std::round(box->y - sum_h - 6 + (h * 0.5f - 5)), 10, 10};

		box->w = std::max(w, box->w);
	}

	box->w += 25;
	box->h = sum_h + 9;

	// offset if overlapping with the wall
	if (box->x + box->w + 5 > win_x) {
		float dx = box->w - (win_x - box->x) + 5;
		box->x -= dx;

		for (int i = 0; i < box->size; i++) {
			box->text_rects[i].x -= dx;
			box->indicator_rects[i].x -= dx;
		}
	}

	if (box->y - box->h - 5 < 0) {
		float dy = box->h - box->y + 5;
		box->y += dy;

		for (int i = 0; i < box->size; i++) {
			box->text_rects[i].y += dy;
			box->indicator_rects[i].y += dy;
		}
	}
}

static inline void setup_inspect_box(SDL_Renderer* renderer, int last, inspect_box* box, TTF_Font* font, SDL_Color color0, SDL_Color color1, SDL_Color color2) {
	float w, h;
	float sum_h = -3;
	SDL_Color color;
	std::string str;

	box->origin_x = box->contents[0].x - last;
	box->origin_y = box->contents[0].y;

	box->w = 0;

	for (int i = 0; i < box->size; i++) {
		point* pt = &box->contents[i];

		if (pt->type == 0) {
			color = color0;
		} else if (pt->type == 1) {
			color = color1;
		} else {
			color = color2;
		}

		if (display_percents) {
			str = format_val(pt->x, 1) + ", " + format_val(*pt->y * 100, decimal_precision) + "%";
		} else {
			str = format_val(pt->x, 1) + ", " + format_val(*pt->y, decimal_precision);
		}

		if (box->textures[i]) {
			SDL_DestroyTexture(box->textures[i]);
		}

		box->textures[i] = create_text_texture(renderer, str, font, color);
		SDL_GetTextureSize(box->textures[i], &w, &h);
		sum_h += h + 3;

		box->text_rects[i] = relative_rect{20, -sum_h - 6, w, h};
		box->indicator_rects[i] = relative_rect{7, -sum_h - 6 + (h * 0.5f - 5), 10, 10};

		box->w = std::max(w, box->w);
	}

	box->w += 25;
	box->h = sum_h + 9;

	w = box->w - 12;
	h = -box->h - 2;
	box->render_rects[0] = relative_rect{8, h, w, 2};
	box->render_rects[1] = relative_rect{8, -4, w, 2};
	box->render_rects[2] = relative_rect{8, h + 2, w, box->h - 4};
	box->render_rects[3] = relative_rect{2, h, 6, 6};
	w = box->w - 4;
	box->render_rects[4] = relative_rect{w, h, 6, 6};
	box->render_rects[5] = relative_rect{w, -8, 6,6};
	box->render_rects[6] = relative_rect{2, -8, 6, 6};
	h += 6;
	box->render_rects[7] = relative_rect{2, h, 6, box->h - 12};
	box->render_rects[8] = relative_rect{w, h, 6, box->h - 12};
}

static inline void render_mouse_box(SDL_Renderer* renderer, mouse_inspect_box* box, SDL_Texture* tl, SDL_Texture* tr, SDL_Texture* br, SDL_Texture* bl, SDL_Texture* lside, SDL_Texture* rside, SDL_Texture* white, SDL_Texture* red, SDL_Texture* blue) {
	float h = box->h - 12;
	float y = box->y - box->h;

	// box
	SDL_FRect rect = {box->x + 8, y - 2, box->w - 12, 2};
	SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
	SDL_RenderFillRect(renderer, &rect);
	rect.y = box->y - 4;
	SDL_RenderFillRect(renderer, &rect);

	SDL_SetRenderDrawColor(renderer, 30, 30, 30, 255);
	rect.y = y; rect.h = box->h - 4;
	SDL_RenderFillRect(renderer, &rect);

	rect = {rect.x - 6, rect.y - 2, 6, 6};
	SDL_RenderTexture(renderer, tl, nullptr, &rect);
	rect.x += box->w - 6;
	SDL_RenderTexture(renderer, tr, nullptr, &rect);
	rect.y += box->h - 6;
	SDL_RenderTexture(renderer, br, nullptr, &rect);
	rect.x = box->x + 2;
	SDL_RenderTexture(renderer, bl, nullptr, &rect);

	rect.y -= h; rect.h = h;
	SDL_RenderTexture(renderer, lside, nullptr, &rect);
	rect.x += box->w - 6;
	SDL_RenderTexture(renderer, rside, nullptr, &rect);

	// text
	for (int i = 0; i < box->size; i++) {
		SDL_RenderTexture(renderer, box->textures[i], nullptr, &box->text_rects[i]);
		if (box->contents[i].type == 0) {
			SDL_RenderTexture(renderer, white, nullptr, &box->indicator_rects[i]);
		} else if (box->contents[i].type == 1) {
			SDL_RenderTexture(renderer, red, nullptr, &box->indicator_rects[i]);
		} else {
			SDL_RenderTexture(renderer, blue, nullptr, &box->indicator_rects[i]);
		}
	}
}

static inline void render_inspect_box(SDL_Renderer* renderer, float sidebar, float spacing, inspect_box* box, SDL_Texture* tl, SDL_Texture* tr, SDL_Texture* br, SDL_Texture* bl, SDL_Texture* lside, SDL_Texture* rside, SDL_Texture* white, SDL_Texture* red, SDL_Texture* blue) {
	float x = std::round(hwin_x + sidebar + zoom * (box->origin_x * spacing + border - cam_x));
	float y = std::round(hwin_y - zoom * (*box->origin_y * (win_y - double_border) + border - cam_y));

	float offset_x = 0;
	float offset_y = 0;

	float border_x = std::round(hwin_x + sidebar + (win_x - sidebar - cam_x) * zoom);
	float border_y = std::round(hwin_y - zoom * (win_y - cam_y));

	if (x + box->w - 3 > border_x) {
		offset_x = border_x - x - box->w - 5;
	}
	if (y - box->h - 3 < border_y) {
		offset_y = (box->h - y + 5) + border_y;
	}

	// box
	SDL_FRect rect;
	relative_rect r = box->render_rects[0];
	rect.x = x + r.dx + offset_x; rect.y = y + r.dy + offset_y; rect.w = r.w, rect.h = r.h;
	SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
	SDL_RenderFillRect(renderer, &rect);
	rect.y = y + box->render_rects[1].dy + offset_y;
	SDL_RenderFillRect(renderer, &rect);

	SDL_SetRenderDrawColor(renderer, 30, 30, 30, 255);
	r = box->render_rects[2];
	rect.y = y + r.dy + offset_y; rect.h = r.h;
	SDL_RenderFillRect(renderer, &rect);

	r = box->render_rects[3];
	rect.x = x + r.dx + offset_x; rect.y = y + r.dy + offset_y; rect.w = r.w; rect.h = r.h;
	SDL_RenderTexture(renderer, tl, nullptr, &rect);
	rect.x = x + box->render_rects[4].dx + offset_x;
	SDL_RenderTexture(renderer, tr, nullptr, &rect);
	rect.y = y + box->render_rects[5].dy + offset_y;
	SDL_RenderTexture(renderer, br, nullptr, &rect);
	rect.x = x + box->render_rects[6].dx + offset_x;
	SDL_RenderTexture(renderer, bl, nullptr, &rect);

	r = box->render_rects[7];
	rect.y = y + r.dy + offset_y; rect.h = r.h;
	SDL_RenderTexture(renderer, lside, nullptr, &rect);
	rect.x = x + box->render_rects[8].dx + offset_x;
	SDL_RenderTexture(renderer, rside, nullptr, &rect);

	// text
	for (int i = 0; i < box->size; i++) {
		r = box->text_rects[i];
		rect.x = x + r.dx + offset_x; rect.y = y + r.dy + offset_y; rect.w = r.w; rect.h = r.h;
		SDL_RenderTexture(renderer, box->textures[i], nullptr, &rect);

		r = box->indicator_rects[i];
		rect.x = x + r.dx + offset_x; rect.y = y + r.dy + offset_y; rect.w = r.w; rect.h = r.h;

		if (box->contents[i].type == 0) {
			SDL_RenderTexture(renderer, white, nullptr, &rect);
		} else if (box->contents[i].type == 1) {
			SDL_RenderTexture(renderer, red, nullptr, &rect);
		} else {
			SDL_RenderTexture(renderer, blue, nullptr, &rect);
		}
	}
}

int main() {
	SDL_Init(SDL_INIT_EVENTS);
	SDL_Init(SDL_INIT_VIDEO);
	TTF_Init();

	threads.reserve(thread_count);

	for (unsigned int i = 0; i < thread_count; i++) {
		threads.emplace_back(execute_thread, i);
	}

	SDL_Delay(200); // COMMENT OUT FOR FINAL COMPILE

	// ---------------------------------------------------------------------------------- INITAL SETUP ---------------------------------------------------------------------------------- 
	SDL_Window* window = SDL_CreateWindow("Binomal Probability Calculator", win_x, win_y, SDL_WINDOW_RESIZABLE);
	SDL_Renderer* renderer = SDL_CreateRenderer(window, nullptr);

	const float pt_radius = 4.0f; // 1 less than value

	SDL_Texture* black_point = create_circle_texture(renderer, pt_radius - 1.0f, 30, 30, 30);
	SDL_Texture* red_point = create_circle_texture(renderer, pt_radius - 1.0f, 251, 25, 43);
	SDL_Texture* blue_point = create_circle_texture(renderer, pt_radius - 1.0f, 34, 67, 240);

	SDL_Color red_pt = {251, 25, 43, 255};
	SDL_Color blue_pt = {44, 87, 250, 255};

	SDL_Texture* white_indicator = create_circle_texture(renderer, 3.3f, 245, 245, 245);
	SDL_Texture* red_indicator = create_circle_texture(renderer, 3.3f, 251, 25, 43);
	SDL_Texture* blue_indicator = create_circle_texture(renderer, 3.3f, 44, 87, 250);

	SDL_FRect point_rect = {0,0,(int)pt_radius * 2,(int)pt_radius * 2};

	TTF_Font* arial13 = TTF_OpenFont("arial.ttf", 13);
	TTF_Font* arial10 = TTF_OpenFont("arial.ttf", 10);

	SDL_Event event;

	calculate_raw_probablilities();
	calculate_n_minus_prob();
	calculate_n_plus_prob();
	calculated_outcomes = raw_probabilities.size();

	int range = (calculated_outcomes - (int)floor(expected)) * 2;
	
	last = 0;
	if (calculated_outcomes - range > 0) {
		last = calculated_outcomes - range;
		raw_probabilities.erase(raw_probabilities.begin(), raw_probabilities.begin() + last);
		n_minus_prob.erase(n_minus_prob.begin(), n_minus_prob.begin() + last);
		n_plus_prob.erase(n_plus_prob.begin(), n_plus_prob.begin() + last);
		calculated_outcomes = raw_probabilities.size();
	}

	std::string current_input;

	SDL_FRect y_axis_rect;
	SDL_FRect x_axis_rect;

	const int open_width = 250;

	SDL_FRect side_bar = {0,0,side_bar_width - 2, win_y};
	SDL_FRect bar_border = {side_bar_width - 2, 0, 2, win_y};
	SDL_FRect open_side_bar = {side_bar_width - 2, 0, open_width, win_y};
	SDL_FRect open_bar_border = {side_bar_width + open_width - 2 , 0, 2, win_y};

	button settings_button = {4, 4, 28, 28};
	settings_button.icon = IMG_LoadTexture(renderer, "textures/menu_button.png");
	settings_button.icon_select = IMG_LoadTexture(renderer, "textures/menu_button_select.png");

	int side_bar_w = side_bar_width;
	float half_bar_w = half_bar;

	bool over_settings = false;
	bool settings_clicked = false;

	SDL_Texture* toggle_off = IMG_LoadTexture(renderer, "textures/toggle_off.png");
	SDL_Texture* toggle_on = IMG_LoadTexture(renderer, "textures/toggle_on.png");

	float w, h;
	SDL_GetTextureSize(toggle_on, &w, &h);

	toggle gridlines_toggle = {renderer, (float)(side_bar_width + open_width - 40), 10, w, h, &display_gridlines};
	toggle percents_toggle = {renderer, (float)(side_bar_width + open_width - 40) , 50, w, h, &display_percents};
	toggle equation_toggle = {renderer, (float)(side_bar_width + open_width - 40), 90, w, h, &display_equation};
	toggle petals_toggle = {renderer, (float)(side_bar_width + open_width - 40), 130, w, h, &calculate_petals};

	bool clicked_toggle = false;

	SDL_Color black = {30,30,30,255};
	SDL_Color white = {245,245,245,255};
	SDL_Color red = {247,32,29,255};

	label gridlines_label = {renderer, (float)(side_bar_width + 5), 10, "Show gridlines", arial13, black};
	label percents_label = {renderer, (float)(side_bar_width + 5), 50, "Display percentages", arial13, black};
	label equation_label = {renderer, (float)(side_bar_width + 5), 90, "Show normal curve", arial13, black};
	label petal_label = {renderer, (float)(side_bar_width + 5), 130, "Calculate using petals", arial13, black};

	SDL_FRect divider_1 = {side_bar_width + 5, 35, open_width - 15, 2};
	SDL_FRect divider_2 = {side_bar_width + 5, 75, open_width - 15, 2};
	SDL_FRect divider_3 = {side_bar_width + 5, 115, open_width - 15, 2};
	SDL_FRect divider_4 = {side_bar_width + 5, 155, open_width - 15, 2};

	SDL_Texture* textbox = IMG_LoadTexture(renderer, "textures/textbox.png");
	float textbox_x = side_bar_width + 10;
	float textbox_y = 207;

	SDL_GetTextureSize(textbox, &w, &h);
	SDL_FRect textbox_rect = {textbox_x, textbox_y, w, h};

	SDL_Texture* input_text = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ABGR32, SDL_TEXTUREACCESS_STATIC, 1, 1);

	double* dropdown_opt1 = &num_trials;
	double* dropdown_opt2 = &chance;
	double* dropdown_opt3 = &threshold;
	int* dropdown_opt4 = &decimal_precision;
	int* dropdown_opt5 = &equation_resolution;

	SDL_Texture* dropdown_closed = IMG_LoadTexture(renderer, "textures/dropdown_closed.png");
	SDL_Texture* dropdown_open = IMG_LoadTexture(renderer, "textures/dropdown_open.png");
	SDL_Texture* dropdown_extension = IMG_LoadTexture(renderer, "textures/dropdown_extension.png");

	SDL_GetTextureSize(dropdown_closed, &w, &h);

	SDL_FRect dropdown_rect = {side_bar_width + 10, textbox_y + 40, w, h};
	SDL_FRect extension_rect = {dropdown_rect.x, dropdown_rect.y, w, h};

	float increment = dropdown_rect.h - 2;

	std::vector<SDL_Texture*> dropdown_displays = {
		create_text_texture(renderer, "Trials = " + format_val(*dropdown_opt1, 20), arial13, white),
		create_text_texture(renderer, "Chance success = " + format_val(*dropdown_opt2, 20), arial13, white),
		create_text_texture(renderer, "Threshold = " + format_val(*dropdown_opt3, 20), arial13, white),
		create_text_texture(renderer, "Decimal precision = " + format_val(*dropdown_opt4, 20), arial13, white),
		create_text_texture(renderer, "Normal curve resolution = " + format_val(*dropdown_opt5, 20), arial13, white)
	};

	const int dropdown_limit = 205;

	std::string dropdown_str;

	SDL_Texture* assignment_error = create_text_texture(renderer, "Input could not be converted to a numerical value.", arial10, red);
	SDL_GetTextureSize(assignment_error, &w, &h);
	SDL_FRect assign_error_rect = {textbox_x + 1, textbox_y - 14, w, h};
	int assign_error_clock = 0;

	std::vector<SDL_FRect> dropdown_display_rects;

	for (int i = 0; i < 5; i++) {
		SDL_GetTextureSize(dropdown_displays[i], &w, &h);
		dropdown_display_rects.emplace_back(SDL_FRect{dropdown_rect.x + 5, dropdown_rect.y + 6 + (extension_rect.h - 2) * i, w, h});
	}

	bool dropdown_bool = false;
	bool dropdown_clicked = false;
	int selected_var = 0;

	button recalculate_button = {(int)side_bar_width + 10.0f, textbox_y - 36.0f, 85, 20};
	recalculate_button.icon = IMG_LoadTexture(renderer, "textures/recalculate_button.png");
	recalculate_button.icon_select = IMG_LoadTexture(renderer, "textures/recalculate_button_select.png");

	bool over_recalculate = false;
	bool recalc_button = false;

	SDL_FRect horizontal_lines_rect = {0,0, 0, 1};
	SDL_FRect vertical_lines_rect = {0,0,1,0};

	std::vector<SDL_Texture*> vertical_labels = create_vertical_labels(renderer, arial10, black);
	std::vector<SDL_FRect> vert_label_rects;

	for (auto& lbl : vertical_labels) {
		SDL_GetTextureSize(lbl, &w, &h);
		vert_label_rects.emplace_back(SDL_FRect{0,0,w,h});
	}

	std::vector<SDL_Texture*> horizontal_labels = create_horizontal_labels(renderer, last, arial10, black);
	std::vector<SDL_FRect> hor_label_rects;

	for (auto& lbl : horizontal_labels) {
		SDL_GetTextureSize(lbl, &w, &h);
		hor_label_rects.emplace_back(SDL_FRect{0,0,w,h});
	}

	bool textbox_clicked = false;

	SDL_FRect input_rect = {side_bar_width + 15, textbox_y + 6, 0,0};
	SDL_FRect caret_rect = {0, textbox_y + 3, 1, 18};
	int caret_clock = 0;
	int caret_index = 0;
	int caret_begin = (int)fps_limit / 2;

	SDL_Texture* box_tl = IMG_LoadTexture(renderer, "textures/tl.png");
	SDL_Texture* box_tr = IMG_LoadTexture(renderer, "textures/tr.png");
	SDL_Texture* box_br = IMG_LoadTexture(renderer, "textures/br.png");
	SDL_Texture* box_bl = IMG_LoadTexture(renderer, "textures/bl.png");
	SDL_Texture* box_lside = IMG_LoadTexture(renderer, "textures/lside.png");
	SDL_Texture* box_rside = IMG_LoadTexture(renderer, "textures/rside.png");

	inspect_point mouse_inspect[5];
	int mouse_inspect_size = 0;
	std::vector<inspect_box> inspect_boxes;
	std::vector<double*> in_box;
	mouse_inspect_box mouse_inspect_box;
	bool box_added = false;

	bool equation_changed = true;

	button clear_button = {side_bar_width + 104.0f, textbox_y - 36.0f, 85, 20};
	clear_button.icon = IMG_LoadTexture(renderer, "textures/clear_button.png");
	clear_button.icon_select = IMG_LoadTexture(renderer, "textures/clear_button_select.png");

	bool clear_button_clicked = false;
	bool over_clear = false;

	SDL_Texture* average_display = create_text_texture(renderer, "Average: " + format_val(expected, decimal_precision), arial13, black);
	SDL_GetTextureSize(average_display, &w, &h);
	SDL_FRect avg_disp_rect = {win_x - w - 5, 5, w, h};

	SDL_Texture* info_icon = IMG_LoadTexture(renderer, "textures/information_icon.png");
	SDL_Texture* info_select = IMG_LoadTexture(renderer, "textures/information_icon_select.png");
	SDL_GetTextureSize(info_icon, &w, &h);
	SDL_FRect info_icon_rect = {side_bar_width + open_width - w - 10, textbox_y - h - 15, w, h};

	float info_icon_x = info_icon_rect.x + info_icon_rect.w * 0.5;
	float info_icon_y = info_icon_rect.y + info_icon_rect.h * 0.5;
	float info_icon_r2 = std::pow(info_icon_rect.w * 0.5, 2);

	SDL_Texture* info_box = IMG_LoadTexture(renderer, "textures/info_box.png");
	SDL_GetTextureSize(info_box, &w, &h);
	SDL_FRect info_box_rect = {side_bar_width + open_width + 10, 10, w, h};

	float box_height = h;

	SDL_FRect info_panel_rect = {0, 0, w, h-4};

	SDL_Texture* info_panel = IMG_LoadTexture(renderer, "textures/info_panel.png");
	SDL_GetTextureSize(info_panel, &w, &h);
	info_panel_rect.w = w;
	SDL_FRect info_prj_rect = {side_bar_width + open_width + 15, 12, w, info_panel_rect.h};

	float ip_scroll_y = 0;
	const float max_ip_scroll_y = h - box_height + 4;

	float bar_travel_length = box_height - 6;
	int scroll_bar_height = std::max(bar_travel_length - max_ip_scroll_y, 50.0f);
	float bar_min_y = info_box_rect.y + 3;
	float bar_max_y = bar_min_y + bar_travel_length - scroll_bar_height;

	SDL_Texture* scroll_bar = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_TARGET, 7, scroll_bar_height);

	SDL_SetRenderTarget(renderer, scroll_bar);
	SDL_SetRenderDrawColor(renderer, 175, 175, 175, 255);
	SDL_FRect rect = SDL_FRect{0, 3, 7, (float)scroll_bar_height - 6};
	SDL_RenderFillRect(renderer, &rect);

	float r = 3.5f;

	for (int dx = 0; dx < 2*r; dx++) {
		for (int dy = 0; dy < 3; dy++) {
			float x = dx - r + 0.5f;
			float y = dy - r + 0.5f;
			float dist = std::sqrt(x * x + y * y);

			if (dist < r) {
				if (dist > r - 1.0f) {
					SDL_SetRenderDrawColor(renderer, 175, 175, 175, (Uint8)std::round((r-dist) * 255.0f));
				} else {
					SDL_SetRenderDrawColor(renderer, 175, 175, 175, 255);
				}

				SDL_RenderPoint(renderer, dx, dy);
			}
		}
	}

	for (int dx = 0; dx < 2 * r; dx++) {
		for (int dy = scroll_bar_height - 3; dy < scroll_bar_height; dy++) {
			float x = dx - r + 0.5f;
			float y = dy + r + 0.5f - scroll_bar_height;
			float dist = std::sqrt(x * x + y * y);

			if (dist < r) {
				if (dist > r - 1.0f) {
					SDL_SetRenderDrawColor(renderer, 175, 175, 175, (Uint8)std::round((r - dist) * 255.0f));
				} else {
					SDL_SetRenderDrawColor(renderer, 175, 175, 175, 255);
				}

				SDL_RenderPoint(renderer, dx, dy);
			}
		}
	}

	SDL_FRect scroll_bar_rect = {info_box_rect.x + info_box_rect.w - 10,  bar_min_y + (bar_max_y - bar_min_y) * (ip_scroll_y / max_ip_scroll_y), 7, scroll_bar_height};

	bool scroll_bar_clicked = false;

	float bar_clicked_y;

	SDL_SetRenderTarget(renderer, nullptr);

	bool display_info = false;
	bool info_clicked = false;

	double avg_trials = convert_trials_to_petals(petals, chance);

	SDL_Texture* average_trials_dislay = create_text_texture(renderer, format_val(petals, decimal_precision) + " petals is " + format_val(avg_trials, decimal_precision) + " trials on average", arial10, black);
	SDL_GetTextureSize(average_trials_dislay, &w, &h);
	SDL_FRect average_trials_rect = {side_bar_width + 3, 3, w, h};

	bool display_avg_trials = false;

	float clicked_x = 0;
	float clicked_y = 0;
	float clicked_cam_x = 0;
	float clicked_cam_y = 0;

	bool continue_drag = true;

	double inv_zoom = 1.0 / zoom;
	double zoom_factor;
	float hscreen_x = (hwin_x - half_bar_w) * inv_zoom;
	float hscreen_y = hwin_y * inv_zoom;

	float cam_x_min = hscreen_x + half_bar_w * (float)inv_zoom;
	float cam_x_max = win_x - hscreen_x - side_bar_w + half_bar_w * (float)inv_zoom;
	float cam_y_min = hscreen_y;
	float cam_y_max = win_y - hscreen_y;

	double last_zoom;
	double max_zoom = 2000.0;

	double spacing = (win_x - 2.0f * border - side_bar_width) / (double)(calculated_outcomes - 1.0f);
	double res_spacing;
	float x, y;
	float dist;
	float border_cam_x_dx, border_cam_y_dy;
	float range_x = win_x - double_border;
	float range_y = win_y - double_border;

	const float inv_freq = 1000.0f / SDL_GetPerformanceFrequency();

	float frame_start;
	float frame_time;

	SDL_StartTextInput(window);

	while (active) {
		frame_start = SDL_GetPerformanceCounter() * inv_freq;

		SDL_SetRenderTarget(renderer, nullptr);
		SDL_SetRenderDrawColor(renderer, 245, 245, 245, 255);
		SDL_RenderClear(renderer);

		// ---------------------------------------------------------------------------------- EVENTS CHECK ---------------------------------------------------------------------------------- 
		while (SDL_PollEvent(&event)) {
			if (event.type == SDL_EVENT_QUIT) {
				active = false;
			} else {
				switch (event.type) {
				case SDL_EVENT_MOUSE_MOTION:
					mouse_x = event.motion.x;
					mouse_y = event.motion.y;
					break;

				case SDL_EVENT_MOUSE_WHEEL:
					if (display_info) {
						if ((mouse_x > info_box_rect.x && mouse_x < info_box_rect.x + info_box_rect.w) && (mouse_y > info_box_rect.y && mouse_y < info_box_rect.y + info_box_rect.h)) {
							ip_scroll_y += event.wheel.y * 25;

							ip_scroll_y = std::clamp(ip_scroll_y, 0.0f, max_ip_scroll_y);

							break;
						}
					}

					if (mouse_x > side_bar_w) {
						zoom_factor = std::pow(1.1, event.wheel.y);

						last_zoom = zoom;

						zoom = std::clamp(zoom * zoom_factor, 1.0, max_zoom);
						inv_zoom = 1.0 / zoom;

						if (zoom != last_zoom) {
							hscreen_x = (hwin_x - half_bar_w) * inv_zoom;
							hscreen_y = hwin_y * inv_zoom;

							cam_x += (mouse_x - hwin_x - side_bar_w) * (zoom_factor - 1.0) * inv_zoom;
							cam_y -= (mouse_y - hwin_y) * (zoom_factor - 1.0) * inv_zoom;

							cam_x_min = hscreen_x + half_bar_w * (float)inv_zoom;
							cam_x_max = win_x - hscreen_x - side_bar_w + half_bar_w * (float)inv_zoom;
							cam_y_min = hscreen_y;
							cam_y_max = win_y - hscreen_y;

							cam_x = std::clamp(cam_x, cam_x_min, cam_x_max);
							cam_y = std::clamp(cam_y, cam_y_min, cam_y_max);
						}
					}

					break;

				case SDL_EVENT_MOUSE_BUTTON_DOWN:
					if (event.button.button == SDL_BUTTON_LEFT) {
						lmb = true;
					}
					break;

				case SDL_EVENT_MOUSE_BUTTON_UP:
					if (event.button.button == SDL_BUTTON_LEFT) {
						lmb = false;
					}
					break;

				case SDL_EVENT_WINDOW_RESIZED:
					win_x = event.window.data1;
					win_y = event.window.data2;

					range_x = win_x - double_border;
					range_y = win_y - double_border;

					hwin_x = win_x * 0.5f;
					hwin_y = win_y * 0.5f;

					spacing = (range_x - side_bar_w) / (double)(calculated_outcomes - 1);
					res_spacing = spacing / equation_resolution;

					hscreen_x = (hwin_x - half_bar_w) * inv_zoom;
					hscreen_y = hwin_y * inv_zoom;

					cam_x_min = hscreen_x + half_bar_w * (float)inv_zoom;
					cam_x_max = win_x - hscreen_x - side_bar_w + half_bar_w * (float)inv_zoom;
					cam_y_min = hscreen_y;
					cam_y_max = win_y - hscreen_y;

					cam_x = std::clamp(cam_x, cam_x_min, cam_x_max);
					cam_y = std::clamp(cam_y, cam_y_min, cam_y_max);

					side_bar.h = win_y;
					bar_border.h = win_y;
					open_side_bar.h = win_y;
					open_bar_border.h = win_y;

					break;

				case SDL_EVENT_TEXT_INPUT:
					if (input_active) {
						current_input.insert(caret_index, event.text.text);
						caret_index += 1;

						caret_clock = caret_begin;
					}
					break;

				case SDL_EVENT_KEY_DOWN:
					switch (event.key.key) {
					case SDLK_BACKSPACE:
						if (input_active && !current_input.empty()) {
							bool ctrl = (event.key.mod & SDL_KMOD_CTRL) != 0;

							if (ctrl) {
								int initial = caret_index;

								if (caret_index != 0 && current_input[caret_index - 1] == ' ') {
									while (caret_index != 0 && current_input[caret_index - 1] == ' ') {
										caret_index -= 1;
									}
								}

								if (!current_input.empty()) {
									while (caret_index != 0 && current_input[caret_index - 1] != ' ') {
										caret_index -= 1;
									}
								}

								current_input.erase(current_input.begin() + caret_index, current_input.begin() + initial);
							} else {
								current_input.erase(current_input.begin() + caret_index - 1);
								caret_index -= 1;
							}

							caret_clock = caret_begin;
						}

						break;

					case SDLK_DELETE:
						if (input_active && !current_input.empty()) {
							bool ctrl = (event.key.mod & SDL_KMOD_CTRL) != 0;

							if (ctrl) {
								int i = 0;

								if (current_input[caret_index + i] == ' ' && caret_index + i != current_input.size()) {
									while (current_input[caret_index + i] == ' ') {
										i++;
									}
								}

								if (!current_input.empty()) {
									while (current_input[caret_index + i] != ' ' && caret_index + i != current_input.size()) {
										i++;
									}
								}

								current_input.erase(current_input.begin() + caret_index, current_input.begin() + caret_index + i);
							} else {
								current_input.erase(current_input.begin() + caret_index);
							}

							caret_clock = caret_begin;
						}

						break;

					case SDLK_RETURN:
						if (input_active) {

							int i = 0;
							while (!current_input.empty() && current_input[i] == ' ') {
								i++;
							}

							if (i > 0) {
								current_input.erase(current_input.begin(), current_input.begin() + i);
							}

							while (!current_input.empty() && current_input.back() == ' ') {
								current_input.pop_back();
							}

							if (valid_number(current_input)) {

								assign_error_clock = 0;

								switch (selected_var) {
								case 0: // ----------------- TRIALS ----------------- 
									if (calculate_petals) {
										petals = std::max(std::stod(current_input), 0.0);

										dropdown_str = format_val(petals, 20);

										SDL_DestroyTexture(dropdown_displays[0]);
										dropdown_displays[0] = create_text_texture(renderer, "Petals = " + dropdown_str, arial13, white);
										SDL_GetTextureSize(dropdown_displays[0], &w, &h);

										if (w > dropdown_limit) {
											dropdown_str.pop_back();

											dropdown_str += "...";

											SDL_DestroyTexture(dropdown_displays[0]);
											dropdown_displays[0] = create_text_texture(renderer, "Petals = " + dropdown_str, arial13, white);
											SDL_GetTextureSize(dropdown_displays[0], &w, &h);

											while (w > dropdown_limit) {
												dropdown_str.erase(dropdown_str.end() - 4);

												SDL_DestroyTexture(dropdown_displays[0]);
												dropdown_displays[0] = create_text_texture(renderer, "Petals = " + dropdown_str, arial13, white);
												SDL_GetTextureSize(dropdown_displays[0], &w, &h);
											}
										}

									} else {
										*dropdown_opt1 = std::max(std::stod(current_input), 0.0);

										dropdown_str = format_val(*dropdown_opt1, 20);

										SDL_DestroyTexture(dropdown_displays[0]);
										dropdown_displays[0] = create_text_texture(renderer, "Trials = " + dropdown_str, arial13, white);
										SDL_GetTextureSize(dropdown_displays[0], &w, &h);

										if (w > dropdown_limit) {
											dropdown_str.pop_back();

											dropdown_str += "...";

											SDL_DestroyTexture(dropdown_displays[0]);
											dropdown_displays[0] = create_text_texture(renderer, "Trials = " + dropdown_str, arial13, white);
											SDL_GetTextureSize(dropdown_displays[0], &w, &h);

											while (w > dropdown_limit) {
												dropdown_str.erase(dropdown_str.end() - 4);

												SDL_DestroyTexture(dropdown_displays[0]);
												dropdown_displays[0] = create_text_texture(renderer, "Trials = " + dropdown_str, arial13, white);
												SDL_GetTextureSize(dropdown_displays[0], &w, &h);
											}
										}
									}

									dropdown_display_rects[0].w = w;
									dropdown_display_rects[0].h = h;

									break;

								case 1: // ----------------- CHANCE ----------------- 
									SDL_DestroyTexture(dropdown_displays[1]);

									if (display_percents) {
										*dropdown_opt2 = std::clamp(std::stod(current_input), 0.0, 100.0);

										dropdown_str = format_val(*dropdown_opt2, 20) + "%";
									} else {
										*dropdown_opt2 = std::clamp(std::stod(current_input), 0.0, 1.0);

										dropdown_str = format_val(*dropdown_opt2, 20);
									}

									dropdown_displays[1] = create_text_texture(renderer, "Chance success = " + dropdown_str, arial13, white);
									SDL_GetTextureSize(dropdown_displays[1], &w, &h);

									if (w > dropdown_limit) {
										if (display_percents) {
											dropdown_str.pop_back(); dropdown_str.pop_back();

											dropdown_str += "...%";
										} else {
											dropdown_str.pop_back();

											dropdown_str += "...";
										}

										SDL_DestroyTexture(dropdown_displays[1]);
										dropdown_displays[1] = create_text_texture(renderer, "Chance success = " + dropdown_str, arial13, white);
										SDL_GetTextureSize(dropdown_displays[1], &w, &h);

										int last_digit_offset = 4;

										if (display_percents) {
											last_digit_offset = 5;
										}

										while (w > dropdown_limit) {
											dropdown_str.erase(dropdown_str.end() - last_digit_offset);

											SDL_DestroyTexture(dropdown_displays[1]);
											dropdown_displays[1] = create_text_texture(renderer, "Chance success = " + dropdown_str, arial13, white);
											SDL_GetTextureSize(dropdown_displays[1], &w, &h);
										}
									}

									dropdown_display_rects[1].w = w;
									dropdown_display_rects[1].h = h;

									if (display_percents) {
										*dropdown_opt2 /= 100;
									}

									break;

								case 2: // ----------------- THRESHOLD ----------------- 
									SDL_DestroyTexture(dropdown_displays[2]);

									if (display_percents) {
										*dropdown_opt3 = std::clamp(std::stod(current_input), 0.0, 100.0);

										dropdown_str = format_val(*dropdown_opt3, 20) + "%";
									} else {
										*dropdown_opt3 = std::clamp(std::stod(current_input), 0.0, 1.0);

										dropdown_str = format_val(*dropdown_opt3, 20);
									}

									dropdown_displays[2] = create_text_texture(renderer, "Threshold = " + dropdown_str, arial13, white);
									SDL_GetTextureSize(dropdown_displays[2], &w, &h);

									if (w > dropdown_limit) {
										if (display_percents) {
											dropdown_str.pop_back(); dropdown_str.pop_back();

											dropdown_str += "...%";
										} else {
											dropdown_str.pop_back();

											dropdown_str += "...";
										}

										SDL_DestroyTexture(dropdown_displays[2]);
										dropdown_displays[2] = create_text_texture(renderer, "Threshold = " + dropdown_str, arial13, white);
										SDL_GetTextureSize(dropdown_displays[2], &w, &h);

										int last_digit_offset = 4;

										if (display_percents) {
											last_digit_offset = 5;
										}

										while (w > dropdown_limit) {
											dropdown_str.erase(dropdown_str.end() - last_digit_offset);

											SDL_DestroyTexture(dropdown_displays[2]);
											dropdown_displays[2] = create_text_texture(renderer, "Threshold = " + dropdown_str, arial13, white);
											SDL_GetTextureSize(dropdown_displays[2], &w, &h);
										}
									}

									dropdown_display_rects[2].w = w;
									dropdown_display_rects[2].h = h;

									if (display_percents) {
										*dropdown_opt3 /= 100;
									}

									break;

								case 3: // ----------------- DECIMAL PRECISION ----------------- 
									*dropdown_opt4 = std::clamp((int)std::round(std::stod(current_input)), 1, 16);

									SDL_DestroyTexture(dropdown_displays[3]);
									dropdown_displays[3] = create_text_texture(renderer, "Decimal precision = " + format_val(*dropdown_opt4, 20), arial13, white);
									SDL_GetTextureSize(dropdown_displays[3], &w, &h);
									dropdown_display_rects[3].w = w;
									dropdown_display_rects[3].h = h;

									for (auto& box : inspect_boxes) {
										for (auto& tx : box.textures) {
											SDL_DestroyTexture(tx);
										}
									}

									for (auto& box : inspect_boxes) {
										setup_inspect_box(renderer, last, &box, arial13, white, red_pt, blue_pt);
									}

									SDL_DestroyTexture(average_display);
									average_display = create_text_texture(renderer, "Average: " + format_val(expected, decimal_precision), arial13, black);
									SDL_GetTextureSize(average_display, &w, &h);

									avg_disp_rect = {win_x - w - 5, 5, w, h};

									if (calculate_petals) {
										SDL_DestroyTexture(average_trials_dislay);
										average_trials_dislay = create_text_texture(renderer, format_val(petals, decimal_precision) + " petals is " + format_val(avg_trials, decimal_precision) + " trials on average", arial10, black);
										SDL_GetTextureSize(average_trials_dislay, &w, &h);
										average_trials_rect = {(float)side_bar_w + 3, 3, w, h};
									}

									break;

								case 4: // ----------------- NORMAL CURVE RES ----------------- 
									*dropdown_opt5 = std::clamp((int)std::round(std::stod(current_input)), 1, 1000);

									SDL_DestroyTexture(dropdown_displays[4]);
									dropdown_displays[4] = create_text_texture(renderer, "Normal curve resolution = " + format_val(*dropdown_opt5, 20), arial13, white);
									SDL_GetTextureSize(dropdown_displays[4], &w, &h);
									dropdown_display_rects[4].w = w;
									dropdown_display_rects[4].h = h;

									if (display_equation) {
										approximate_raw_probs();
										approximate_n_minus_probs();
										approximate_n_plus_probs();


										res_spacing = spacing / equation_resolution;

										equation_changed = true;
									}

									break;
								}
							} else {
								assign_error_clock = fps_limit * 3;
							}

							current_input.clear();
							caret_index = 0;
						}
						break;

					case SDLK_ESCAPE:
						display_settings = !display_settings;

						if (display_settings) {
							side_bar_w = side_bar_width + open_width + 2;
							half_bar_w = side_bar_w / 2.0f;
						} else {
							side_bar_w = side_bar_width;
							half_bar_w = half_bar;

							display_info = false;
						}

						spacing = (range_x - side_bar_w) / (double)(calculated_outcomes - 1);
						res_spacing = spacing / equation_resolution;
						hscreen_x = (hwin_x - half_bar_w) * inv_zoom;

						cam_x_min = hscreen_x + half_bar_w * (float)inv_zoom;
						cam_x_max = win_x - hscreen_x - side_bar_w + half_bar_w * (float)inv_zoom;

						cam_x = std::clamp(cam_x, cam_x_min, cam_x_max);

						average_trials_rect.x = side_bar_w + 3;

						break;

					case SDLK_LEFT:
						if (input_active && !current_input.empty()) {
							bool ctrl = (event.key.mod & SDL_KMOD_CTRL) != 0;

							if (ctrl) {
								if (current_input[caret_index - 1] == ' ' && caret_index != 0) {
									while (current_input[caret_index - 1] == ' ') {
										caret_index -= 1;
									}
								}

								if (caret_index != 0) {
									while (current_input[caret_index - 1] != ' ' && caret_index != 0) {
										caret_index -= 1;
									}
								}
							} else {
								caret_index = std::max(0, caret_index - 1);
							}
							caret_clock = caret_begin;
						}
						break;

					case SDLK_RIGHT:
						if (input_active && !current_input.empty()) {
							bool ctrl = (event.key.mod & SDL_KMOD_CTRL) != 0;

							if (ctrl) {
								if (current_input[caret_index] == ' ' && caret_index != current_input.size()) {
									while (current_input[caret_index] == ' ') {
										caret_index += 1;
									}
								}

								if (caret_index != current_input.size()) {
									while (current_input[caret_index] != ' ' && caret_index != current_input.size()) {
										caret_index += 1;
									}
								}
							} else {
								caret_index = std::min(caret_index + 1, (int)current_input.size());
							}
							caret_clock = caret_begin;
						}
						break;
					}

					break;
				}
			}
		}

		// ---------------------------------------------------------------------------------- SETTINGS MENU INTERACTIONS ---------------------------------------------------------------------------------- 
		if (display_settings && lmb) {
			// toggles
			if (!clicked_toggle) {
				float dx; float dy;

				if (mouse_x > gridlines_toggle.x + 14.5f) {
					dx = gridlines_toggle.x + 21 - mouse_x;
				} else {
					dx = gridlines_toggle.x + 8 - mouse_x;
				}
				dy = gridlines_toggle.y + 8 - mouse_y;

				dist = dx * dx + dy * dy;

				if (dist < 55.0f || (dx < 0 && mouse_x > gridlines_toggle.x + 14.5f && mouse_y > gridlines_toggle.y && mouse_y < gridlines_toggle.y + gridlines_toggle.rect.h) || (dx > 0 && mouse_x > gridlines_toggle.x + 14.5f && mouse_y > gridlines_toggle.y && mouse_y < gridlines_toggle.y + gridlines_toggle.rect.h)) {
					*gridlines_toggle.state = !*gridlines_toggle.state;

					clicked_toggle = true;
				} else {
					if (mouse_x > percents_toggle.x + 14.5f) {
						dx = percents_toggle.x + 21 - mouse_x;
					} else {
						dx = percents_toggle.x + 8 - mouse_x;
					}
					dy = percents_toggle.y + 8 - mouse_y;

					dist = dx * dx + dy * dy;

					if (dist < 55.0f || (dx < 0 && mouse_x > percents_toggle.x + 14.5f && mouse_y > percents_toggle.y && mouse_y < percents_toggle.y + percents_toggle.rect.h) || (dx > 0 && mouse_x > percents_toggle.x + 14.5f && mouse_y > percents_toggle.y && mouse_y < percents_toggle.y + percents_toggle.rect.h)) {
						*percents_toggle.state = !*percents_toggle.state;

						SDL_DestroyTexture(dropdown_displays[1]);

						if (display_percents) {
							*dropdown_opt2 *= 100.0;

							dropdown_str = format_val(*dropdown_opt2, 20) + "%";
						} else {
							dropdown_str = format_val(*dropdown_opt2, 20);
						}

						dropdown_displays[1] = create_text_texture(renderer, "Chance success = " + dropdown_str, arial13, white);
						SDL_GetTextureSize(dropdown_displays[1], &w, &h);

						if (w > dropdown_limit) {
							if (display_percents) {
								dropdown_str.pop_back(); dropdown_str.pop_back();

								dropdown_str += "...%";
							} else {
								dropdown_str.pop_back();

								dropdown_str += "...";
							}

							SDL_DestroyTexture(dropdown_displays[1]);
							dropdown_displays[1] = create_text_texture(renderer, "Chance success = " + dropdown_str, arial13, white);
							SDL_GetTextureSize(dropdown_displays[1], &w, &h);

							int last_digit_offset = 4;

							if (display_percents) {
								last_digit_offset = 5;
							}

							while (w > dropdown_limit) {
								dropdown_str.erase(dropdown_str.end() - last_digit_offset);

								SDL_DestroyTexture(dropdown_displays[1]);
								dropdown_displays[1] = create_text_texture(renderer, "Chance success = " + dropdown_str, arial13, white);
								SDL_GetTextureSize(dropdown_displays[1], &w, &h);
							}
						}

						dropdown_display_rects[1].w = w;
						dropdown_display_rects[1].h = h;

						SDL_DestroyTexture(dropdown_displays[2]);

						if (display_percents) {
							*dropdown_opt3 *= 100.0;

							dropdown_str = format_val(*dropdown_opt3, 20) + "%";
						} else {
							dropdown_str = format_val(*dropdown_opt3, 20);
						}

						dropdown_displays[2] = create_text_texture(renderer, "Threshold = " + dropdown_str, arial13, white);
						SDL_GetTextureSize(dropdown_displays[2], &w, &h);

						if (w > dropdown_limit) {
							if (display_percents) {
								dropdown_str.pop_back(); dropdown_str.pop_back();

								dropdown_str += "...%";
							} else {
								dropdown_str.pop_back();

								dropdown_str += "...";
							}

							SDL_DestroyTexture(dropdown_displays[2]);
							dropdown_displays[2] = create_text_texture(renderer, "Threshold = " + dropdown_str, arial13, white);
							SDL_GetTextureSize(dropdown_displays[2], &w, &h);

							int last_digit_offset = 4;

							if (display_percents) {
								last_digit_offset = 5;
							}

							while (w > dropdown_limit) {
								dropdown_str.erase(dropdown_str.end() - last_digit_offset);

								SDL_DestroyTexture(dropdown_displays[2]);
								dropdown_displays[2] = create_text_texture(renderer, "Threshold = " + dropdown_str, arial13, white);
								SDL_GetTextureSize(dropdown_displays[2], &w, &h);
							}
						}

						dropdown_display_rects[2].w = w;
						dropdown_display_rects[2].h = h;

						if (display_percents) {
							*dropdown_opt2 /= 100.0;
							*dropdown_opt3 /= 100.0;
						}

						for (auto& lbl : vertical_labels) {
							SDL_DestroyTexture(lbl);
							lbl = nullptr;
						}

						for (auto& box : inspect_boxes) {
							for (auto& tx : box.textures) {
								SDL_DestroyTexture(tx);
								tx = nullptr;
							}
						}

						for (auto& box : inspect_boxes) {
							setup_inspect_box(renderer, last, &box, arial13, white, red_pt, blue_pt);
						}

						vertical_labels = create_vertical_labels(renderer, arial10, black);

						vert_label_rects.clear();

						for (int i = 0; i < vertical_labels.size(); i++) {
							SDL_GetTextureSize(vertical_labels[i], &w, &h);
							vert_label_rects.emplace_back(SDL_FRect{0,0,w,h});
						}

						clicked_toggle = true;
					} else {
						if (mouse_x > equation_toggle.x + 14.5f) {
							dx = equation_toggle.x + 21 - mouse_x;
						} else {
							dx = equation_toggle.x + 8 - mouse_x;
						}
						dy = equation_toggle.y + 8 - mouse_y;

						dist = dx * dx + dy * dy;

						if (dist < 55.0f || (dx < 0 && mouse_x > equation_toggle.x + 14.5f && mouse_y > equation_toggle.y && mouse_y < equation_toggle.y + equation_toggle.rect.h) || (dx > 0 && mouse_x > equation_toggle.x + 14.5f && mouse_y > equation_toggle.y && mouse_y < equation_toggle.y + equation_toggle.rect.h)) {

							*equation_toggle.state = !*equation_toggle.state;

							clicked_toggle = true;

							selected_var = std::min(selected_var, (int)dropdown_displays.size() - 2);

							if (display_equation && equation_changed) {
								approximate_raw_probs();
								approximate_n_minus_probs();
								approximate_n_plus_probs();

								res_spacing = spacing / equation_resolution;

								equation_changed = false;
							}
						} else {
							if (mouse_x > petals_toggle.x + 14.5f) {
								dx = petals_toggle.x + 21 - mouse_x;
							} else {
								dx = petals_toggle.x + 8 - mouse_x;
							}
							dy = petals_toggle.y + 8 - mouse_y;

							dist = dx * dx + dy * dy;

							if (dist < 55.0f || (dx < 0 && mouse_x > petals_toggle.x + 14.5f && mouse_y > petals_toggle.y && mouse_y < petals_toggle.y + petals_toggle.rect.h) || (dx > 0 && mouse_x > petals_toggle.x + 14.5f && mouse_y > petals_toggle.y && mouse_y < petals_toggle.y + petals_toggle.rect.h)) {

								*petals_toggle.state = !*petals_toggle.state;

								
								clicked_toggle = true;
							}
						}
					}
				}
			}

			// textbox
			if ((mouse_x > textbox_rect.x && mouse_x < textbox_rect.x + textbox_rect.w) && (mouse_y > textbox_rect.y && mouse_y < textbox_rect.y + textbox_rect.h)) {
				if (!textbox_clicked) {
					input_active = true;

					caret_clock = caret_begin;
					caret_index = current_input.size();

					textbox_clicked = true;
				}
			} else {
				input_active = false;
			}

			// dropdown
			if (dropdown_bool) {
				float dropdown_height = dropdown_rect.h + (dropdown_displays.size() - 2) * (extension_rect.h - 2);

				if (display_equation) {
					dropdown_height += extension_rect.h - 2;
				}

				if ((mouse_x > dropdown_rect.x && mouse_x < dropdown_rect.x + dropdown_rect.w) && (mouse_y > dropdown_rect.y && mouse_y < dropdown_rect.y + dropdown_height)) {
					if (!dropdown_clicked) {
						dropdown_bool = false;

						if (mouse_y > dropdown_rect.y + dropdown_rect.h - 1 && mouse_y < dropdown_rect.y + dropdown_height) {
							for (int i = 0; i < dropdown_displays.size() - 2; i++) {
								if (mouse_y < dropdown_rect.y + dropdown_rect.h - 1 + (extension_rect.h - 2) * (i + 1)) {
									if (selected_var <= i) {
										selected_var = i + 1;
									} else {
										selected_var = i;
									}
									break;
								}
							}

							if (display_equation) {
								if (mouse_y > dropdown_rect.y + dropdown_height - extension_rect.h - 1) {
									selected_var = dropdown_displays.size() - 1;
								}
							}
						}

						dropdown_clicked = true;
					}
				}
			} else if ((mouse_x > dropdown_rect.x && mouse_x < dropdown_rect.x + dropdown_rect.w) && (mouse_y > dropdown_rect.y && mouse_y < dropdown_rect.y + dropdown_rect.h)) {
				if (!dropdown_clicked) {
					dropdown_bool = true;

					dropdown_clicked = true;
				}
			}

			// recalculate button
			if ((mouse_x > recalculate_button.x && mouse_x < recalculate_button.x + recalculate_button.w) && (mouse_y > recalculate_button.y && mouse_y < recalculate_button.y + recalculate_button.h)) {
				over_recalculate = true;

				if (!recalc_button && lmb) {
					if (calculate_petals) {
						avg_trials = convert_trials_to_petals(petals, chance);

						current_trials = avg_trials;

						calculate_raw_probablilities();
						
						display_avg_trials = true;

					} else {
						current_trials = num_trials;

						calculate_raw_probablilities();

						display_avg_trials = false;
					}
					calculate_n_minus_prob();
					calculate_n_plus_prob();
					calculated_outcomes = raw_probabilities.size();

					int range = (calculated_outcomes - (int)floor(expected)) * 2;

					last = 0;
					if (calculated_outcomes - range > 0) {
						last = calculated_outcomes - range;
						raw_probabilities.erase(raw_probabilities.begin(), raw_probabilities.begin() + last);
						n_minus_prob.erase(n_minus_prob.begin(), n_minus_prob.begin() + last);
						n_plus_prob.erase(n_plus_prob.begin(), n_plus_prob.begin() + last);
						calculated_outcomes = raw_probabilities.size();
					}

					spacing = (range_x - side_bar_w) / (double)(calculated_outcomes - 1);

					if (display_equation) {
						approximate_raw_probs();
						approximate_n_minus_probs();
						approximate_n_plus_probs();

						res_spacing = spacing / equation_resolution;
					}

					if (calculate_petals) {
						SDL_DestroyTexture(average_trials_dislay);
						average_trials_dislay = create_text_texture(renderer, format_val(petals, decimal_precision) + " petals is " + format_val(avg_trials, decimal_precision) + " trials on average", arial10, black);
						SDL_GetTextureSize(average_trials_dislay, &w, &h);
						average_trials_rect = {(float)side_bar_w + 3, 3, w, h};

						dropdown_str = format_val(petals, 20);

						SDL_DestroyTexture(dropdown_displays[0]);
						dropdown_displays[0] = create_text_texture(renderer, "Petals = " + dropdown_str, arial13, white);
						SDL_GetTextureSize(dropdown_displays[0], &w, &h);

						if (w > dropdown_limit) {
							dropdown_str.pop_back();

							dropdown_str += "...";

							SDL_DestroyTexture(dropdown_displays[0]);
							dropdown_displays[0] = create_text_texture(renderer, "Petals = " + dropdown_str, arial13, white);
							SDL_GetTextureSize(dropdown_displays[0], &w, &h);

							while (w > dropdown_limit) {
								dropdown_str.erase(dropdown_str.end() - 4);

								SDL_DestroyTexture(dropdown_displays[0]);
								dropdown_displays[0] = create_text_texture(renderer, "Petals = " + dropdown_str, arial13, white);
								SDL_GetTextureSize(dropdown_displays[0], &w, &h);
							}
						}
					} else {
						dropdown_str = format_val(*dropdown_opt1, 20);

						SDL_DestroyTexture(dropdown_displays[0]);
						dropdown_displays[0] = create_text_texture(renderer, "Trials = " + dropdown_str, arial13, white);
						SDL_GetTextureSize(dropdown_displays[0], &w, &h);

						if (w > dropdown_limit) {
							dropdown_str.pop_back();

							dropdown_str += "...";

							SDL_DestroyTexture(dropdown_displays[0]);
							dropdown_displays[0] = create_text_texture(renderer, "Trials = " + dropdown_str, arial13, white);
							SDL_GetTextureSize(dropdown_displays[0], &w, &h);

							while (w > dropdown_limit) {
								dropdown_str.erase(dropdown_str.end() - 4);

								SDL_DestroyTexture(dropdown_displays[0]);
								dropdown_displays[0] = create_text_texture(renderer, "Trials = " + dropdown_str, arial13, white);
								SDL_GetTextureSize(dropdown_displays[0], &w, &h);
							}
						}
					}

					dropdown_display_rects[0].w = w;
					dropdown_display_rects[0].h = h;

					SDL_DestroyTexture(average_display);
					average_display = create_text_texture(renderer, "Average: " + format_val(expected, decimal_precision), arial13, black);
					SDL_GetTextureSize(average_display, &w, &h);
					avg_disp_rect = {win_x - w - 5, 5, w, h};

					equation_changed = true;

					for (auto& lbl : horizontal_labels) {
						SDL_DestroyTexture(lbl);
						lbl = nullptr;
					}

					horizontal_labels = create_horizontal_labels(renderer, last, arial10, black);

					hor_label_rects.clear();

					for (auto& lbl : horizontal_labels) {
						SDL_GetTextureSize(lbl, &w, &h);
						hor_label_rects.emplace_back(SDL_FRect{0,0,w,h});
					}

					for (auto& box : inspect_boxes) {
						for (auto& tx : box.textures) {
							SDL_DestroyTexture(tx);
							tx = nullptr;
						}
					}

					inspect_boxes.clear();
					in_box.clear();

					recalc_button = true;
				}
			}

			// clear button
			if (inspect_boxes.size() > 0) {
				if ((mouse_x > clear_button.x && mouse_x < clear_button.x + recalculate_button.w) && (mouse_y > clear_button.y && mouse_y < clear_button.y + clear_button.h)) {
					over_clear = true;

					if (!clear_button_clicked) {
						inspect_boxes.clear();
						in_box.clear();
					}
				}
			}
		} else {
			clicked_toggle = false;
			textbox_clicked = false;
			dropdown_clicked = false;
			over_recalculate = false;
			recalc_button = false;
			clear_button_clicked = false;
			over_clear = false;

			// hovering over button checks
			if ((mouse_x > recalculate_button.x && mouse_x < recalculate_button.x + recalculate_button.w) && (mouse_y > recalculate_button.y && mouse_y < recalculate_button.y + recalculate_button.h)) {
				over_recalculate = true;
			}

			if (inspect_boxes.size() > 0) {
				if ((mouse_x > clear_button.x && mouse_x < clear_button.x + recalculate_button.w) && (mouse_y > clear_button.y && mouse_y < clear_button.y + clear_button.h)) {
					over_clear = true;
				}
			}
		}

		// scroll bar
		if (display_info) {
			if (lmb) {
				if (!scroll_bar_clicked && (mouse_x > scroll_bar_rect.x && mouse_x < scroll_bar_rect.x + scroll_bar_rect.w) && (mouse_y > scroll_bar_rect.y && mouse_y < scroll_bar_rect.y + scroll_bar_rect.h)) {
					scroll_bar_clicked = true;

					bar_clicked_y = scroll_bar_rect.y - mouse_y;
				}

			} else {
				scroll_bar_clicked = false;
			}

			if (scroll_bar_clicked) {
				scroll_bar_rect.y = std::round(std::clamp(mouse_y + bar_clicked_y, bar_min_y, bar_max_y));
				ip_scroll_y = (scroll_bar_rect.y - bar_min_y) / (bar_max_y - bar_min_y) * max_ip_scroll_y;

			} else {
				scroll_bar_rect.y = std::round(bar_min_y + (bar_max_y - bar_min_y) * (ip_scroll_y / max_ip_scroll_y));
			}
		}

		// ---------------------------------------------------------------------------------- SETTINGS BUTTON ---------------------------------------------------------------------------------- 
		if (mouse_x > settings_button.x && mouse_x < settings_button.x + settings_button.w && mouse_y > settings_button.y && mouse_y < settings_button.y + settings_button.h) {
			over_settings = true;
			if (lmb) {
				if (!settings_clicked) {
					display_settings = !display_settings;

					if (display_settings) {
						side_bar_w = side_bar_width + open_width + 2;
						half_bar_w = side_bar_w / 2.0f;
					} else {
						side_bar_w = side_bar_width;
						half_bar_w = half_bar;

						display_info = false;
					}

					average_trials_rect.x = side_bar_w + 3;

					spacing = (range_x - side_bar_w) / (double)(calculated_outcomes - 1);
					res_spacing = spacing / equation_resolution;

					hscreen_x = (hwin_x - half_bar_w) * inv_zoom;

					cam_x_min = hscreen_x + half_bar_w * (float)inv_zoom;
					cam_x_max = win_x - hscreen_x - side_bar_w + half_bar_w * (float)inv_zoom;

					cam_x = std::clamp(cam_x, cam_x_min, cam_x_max);

					settings_clicked = true;
				}
			} else {
				settings_clicked = false;
			}
		} else {
			settings_clicked = false;
			over_settings = false;
		}

		// ---------------------------------------------------------------------------------- DRAG MOVEMENT ---------------------------------------------------------------------------------- 
		if (lmb && !scroll_bar_clicked) {
			if (!drag && mouse_x > side_bar_w && !box_added && !(display_info && (mouse_x > info_box_rect.x && mouse_x < info_box_rect.x + info_box_rect.w) && (mouse_y > info_box_rect.y && mouse_y < info_box_rect.y + info_box_rect.h))) {
				clicked_x = mouse_x;
				clicked_y = mouse_y;
				clicked_cam_x = cam_x;
				clicked_cam_y = cam_y;
				drag = true;
			}
		} else {
			drag = false;
		}

		if (drag && !box_added && continue_drag) {
			float dx = clicked_x - mouse_x;
			float dy = clicked_y - mouse_y;

			cam_x = std::clamp(clicked_cam_x + dx * (float)inv_zoom, cam_x_min, cam_x_max);
			cam_y = std::clamp(clicked_cam_y + -dy * (float)inv_zoom, cam_y_min, cam_y_max);
		}

		border_cam_x_dx = border - cam_x;
		border_cam_y_dy = border - cam_y;

		// ---------------------------------------------------------------------------------- RENDER GRIDLINES ---------------------------------------------------------------------------------- 
		// horizontal
		if (display_gridlines) {
			SDL_SetRenderDrawColor(renderer, 222, 222, 222, 255);

			horizontal_lines_rect.x = (float)(hwin_x + side_bar_w + zoom * border_cam_x_dx);
			horizontal_lines_rect.y = (float)(hwin_y - zoom * (win_y - border - cam_y));
			horizontal_lines_rect.w = (float)(zoom * (range_x - side_bar_w));

			float increment = range_y * zoom * 0.1;

			int i = 0;
			while (horizontal_lines_rect.y < 0) {
				horizontal_lines_rect.y += increment;
				i++;
			}

			for (i; i < 10; i++) {
				SDL_RenderFillRect(renderer, &horizontal_lines_rect);
				horizontal_lines_rect.y += increment;

				if (horizontal_lines_rect.y > win_y) {
					break;
				}
			}

			// vertical
			vertical_lines_rect.x = (float)(hwin_x + side_bar_w + zoom * (border_cam_x_dx));
			vertical_lines_rect.y = (float)(hwin_y - zoom * (win_y - border - cam_y));
			vertical_lines_rect.h = (float)(range_y * zoom);

			increment = spacing * zoom;

			int total;

			if (increment < label_spacing) {
				int skip = std::ceil(50.0 / increment);
				increment *= skip;

				total = std::floor((calculated_outcomes - 1) / skip);
			} else {
				total = calculated_outcomes - 1;
			}

			vertical_lines_rect.x += increment;

			i = 0;
			while (vertical_lines_rect.x < 0) {
				vertical_lines_rect.x += increment;
				i++;
			}

			for (i; i < total; i++) {
				SDL_RenderFillRect(renderer, &vertical_lines_rect);

				vertical_lines_rect.x += increment;

				if (vertical_lines_rect.x > win_x) {
					break;
				}
			}
		}

		y_axis_rect = {(float)(hwin_x + side_bar_w + zoom * border_cam_x_dx - 1), (float)(hwin_y - zoom * (win_y - border - cam_y)), 2, (float)(range_y * zoom)};
		x_axis_rect = {(float)(hwin_x + side_bar_w + zoom * border_cam_x_dx - 1), (float)(hwin_y - zoom * border_cam_y_dy), (float)(zoom * (range_x - side_bar_w)) + 1, 2};

		SDL_SetRenderDrawColor(renderer, 30, 30, 30, 255);
		SDL_RenderFillRect(renderer, &y_axis_rect);
		SDL_RenderFillRect(renderer, &x_axis_rect);

		// ---------------------------------------------------------------------------------- RENDER NORMAL CURVE ---------------------------------------------------------------------------------- 
		size_t start = std::max((-hwin_x * inv_zoom - border_cam_x_dx) / spacing + 1.0, 1.0);

		if (display_equation) {
			int size = equation_raw_probs.size();

			// probability of exactly n (black)
			x = hwin_x + side_bar_w + zoom * border_cam_x_dx;
			y = hwin_y - zoom * (equation_raw_probs[start - 1] * range_y + border_cam_y_dy);

			float x2, y2;

			SDL_SetRenderDrawColor(renderer, 30, 30, 30, 255);
			for (size_t i = start; i < size; i++) {
				x2 = hwin_x + side_bar_w + zoom * (res_spacing * i + border_cam_x_dx);
				y2 = hwin_y - zoom * (equation_raw_probs[i] * range_y + border_cam_y_dy);

				SDL_RenderLine(renderer, x, y, x2, y2);

				x = x2;
				y = y2;

				if (x > win_x) {
					break;
				}
			}

			// n- probability (red)
			x = hwin_x + side_bar_w + zoom * border_cam_x_dx;
			y = hwin_y - zoom * (equation_n_minus_probs[0] * range_y + border_cam_y_dy);

			SDL_SetRenderDrawColor(renderer, 251, 25, 43, 255);
			for (size_t i = start; i < size; i++) {
				x2 = hwin_x + side_bar_w + zoom * (res_spacing * i + border_cam_x_dx);
				y2 = hwin_y - zoom * (equation_n_minus_probs[i] * range_y + border_cam_y_dy);

				SDL_RenderLine(renderer, x, y, x2, y2);

				x = x2;
				y = y2;

				if (x > win_x) {
					break;
				}
			}

			// n+ probability (blue)
			x = hwin_x + side_bar_w + zoom * border_cam_x_dx;
			y = hwin_y - zoom * (equation_n_plus_probs[0] * range_y + border_cam_y_dy);

			SDL_SetRenderDrawColor(renderer, 34, 67, 240, 255);
			for (size_t i = start; i < size; i++) {
				x2 = hwin_x + side_bar_w + zoom * (res_spacing * i + border_cam_x_dx);
				y2 = hwin_y - zoom * (equation_n_plus_probs[i] * range_y + border_cam_y_dy);

				SDL_RenderLine(renderer, x, y, x2, y2);

				x = x2;
				y = y2;

				if (x > win_x) {
					break;
				}
			}
		}

		// ---------------------------------------------------------------------------------- RENDER GRAPH LABELS ---------------------------------------------------------------------------------- 
		// vertical
		float lbl_y = (float)(hwin_y - zoom * (border - cam_y));
		float lbl_x = (float)(hwin_x + side_bar_w + zoom * border_cam_x_dx) - 4;
		float space = range_y * zoom * 0.1;

		int i = 0;
		while (lbl_y > win_y) {
			lbl_y -= space;
			i++;
		}

		SDL_FRect* lbl;
		for (i; i < vertical_labels.size(); i++) {
			lbl = &vert_label_rects[i];
			lbl->x = std::round(std::max(lbl_x - lbl->w, (float)side_bar_w + 3));
			lbl->y = std::round(lbl_y - lbl->h * 0.5);

			SDL_RenderTexture(renderer, vertical_labels[i], nullptr, lbl);

			lbl_y -= space;

			if (lbl_y < -12) {
				break;
			}
		}

		// horizontal
		lbl_x = (float)(hwin_x + side_bar_w + zoom * (border_cam_x_dx));
		lbl_y = (float)(hwin_y - zoom * border_cam_y_dy) + 3;

		space = spacing * zoom;

		int total;

		int skip = 1;
		if (space < label_spacing) {
			skip = std::ceil(label_spacing / space);
			space *= skip;

			total = std::floor((calculated_outcomes - 1) / skip) + 1;
		} else {
			total = calculated_outcomes;
		}

		i = 0;
		while (lbl_x < -20) {
			lbl_x += space;
			i++;
		}

		for (i; i < total; i++) {
			lbl = &hor_label_rects[i * skip];
			lbl->x = std::round(lbl_x - lbl->w * 0.5);
			lbl->y = std::round(std::min(lbl_y, win_y - lbl->h - 3));

			SDL_RenderTexture(renderer, horizontal_labels[i * skip], nullptr, lbl);

			lbl_x += space;

			if (lbl_x > win_x) {
				break;
			}
		}

		if (mouse_inspect_size > 0) {
			for (auto& tx : mouse_inspect_box.textures) {
				SDL_DestroyTexture(tx);
				tx = nullptr;
			}
		}

		mouse_inspect_size = 0;

		// ---------------------------------------------------------------------------------- RENDER DATA POINTS ---------------------------------------------------------------------------------- 
		start--;

		// probability of exactly n (black)
		for (size_t i = start; i < calculated_outcomes; i++) {
			x = hwin_x + side_bar_w + zoom * (spacing * i + border_cam_x_dx);

			if (x > win_x) {
				break;
			}

			y = hwin_y - zoom * (raw_probabilities[i] * range_y + border_cam_y_dy);

			if (y > -pt_radius && y < win_y + pt_radius) {
				double dx = mouse_x - x;
				double dy = mouse_y - y;

				double dist = dx * dx + dy * dy;

				if (dist < 100) {
					int j = 0;

					while (j < mouse_inspect_size && dist > mouse_inspect[j].dist) {
						j++;
					}

					int end = std::min(mouse_inspect_size, 4);
					for (int k = end; k > j; k--) {
						mouse_inspect[k] = mouse_inspect[k - 1];
					}

					mouse_inspect[j] = {dist, point{(float)i + last, &raw_probabilities[i], 0}};

					if (mouse_inspect_size < 5) {
						mouse_inspect_size++;
					}
				}

				point_rect.x = std::round(x - pt_radius); point_rect.y = std::round(y - pt_radius);
				SDL_RenderTexture(renderer, black_point, nullptr, &point_rect);
			}
		}

		// n- probability (red)
		for (size_t i = start; i < calculated_outcomes; i++) {
			x = hwin_x + side_bar_w + zoom * (spacing * i + border_cam_x_dx);

			if (x > win_x) {
				break;
			}

			y = hwin_y - zoom * (n_minus_prob[i] * range_y + border_cam_y_dy);

			if (y > -pt_radius && y < win_y + pt_radius) {
				double dx = mouse_x - x;
				double dy = mouse_y - y;

				double dist = dx * dx + dy * dy;

				if (dist < 100) {
					int j = 0;

					while (j < mouse_inspect_size && dist > mouse_inspect[j].dist) {
						j++;
					}

					int end = std::min(mouse_inspect_size, 4);
					for (int k = end; k > j; k--) {
						mouse_inspect[k] = mouse_inspect[k - 1];
					}

					mouse_inspect[j] = {dist, point{(float)i + last, &n_minus_prob[i], 1}};

					if (mouse_inspect_size < 5) {
						mouse_inspect_size++;
					}
				}

				point_rect.x = std::round(x - pt_radius); point_rect.y = std::round(y - pt_radius);
				SDL_RenderTexture(renderer, red_point, nullptr, &point_rect);
			}
		}

		// n+ porbability (blue)
		for (size_t i = start; i < calculated_outcomes; i++) {
			x = hwin_x + side_bar_w + zoom * (spacing * i + border_cam_x_dx);

			if (x > win_x) {
				break;
			}

			y = hwin_y - zoom * (n_plus_prob[i] * range_y + border_cam_y_dy);

			if (y > -pt_radius && y < win_y + pt_radius) {
				double dx = mouse_x - x;
				double dy = mouse_y - y;

				double dist = dx * dx + dy * dy;

				if (dist < 100) {
					int j = 0;

					while (j < mouse_inspect_size && dist > mouse_inspect[j].dist) {
						j++;
					}

					int end = std::min(mouse_inspect_size, 4);
					for (int k = end; k > j; k--) {
						mouse_inspect[k] = mouse_inspect[k - 1];
					}

					mouse_inspect[j] = {dist, point{(float)i + last, &n_plus_prob[i], 2}};

					if (mouse_inspect_size < 5) {
						mouse_inspect_size++;
					}
				}

				point_rect.x = std::round(x - pt_radius); point_rect.y = std::round(y - pt_radius);
				SDL_RenderTexture(renderer, blue_point, nullptr, &point_rect);
			}
		}
		// ---------------------------------------------------------------------------------- RENDER AVERAGE ----------------------------------------------------------------------------------
		SDL_RenderTexture(renderer, average_display, nullptr, &avg_disp_rect);
		
		if (display_avg_trials) {
			SDL_RenderTexture(renderer, average_trials_dislay, nullptr, &average_trials_rect);
		}

		// ---------------------------------------------------------------------------------- MOUSE INSPECTION ---------------------------------------------------------------------------------- 
		if (mouse_inspect_size > 0) {
			if (lmb && !scroll_bar_clicked && !(display_info && (mouse_x > info_box_rect.x && mouse_x < info_box_rect.x + info_box_rect.w) && (mouse_y > info_box_rect.y && mouse_y < info_box_rect.y + info_box_rect.h))) {
				if (!box_added) {
					box_added = true;

					inspect_point near[5];
					int near_size = 1;

					near[0] = mouse_inspect[0];

					float mapped_x = hwin_x + side_bar_w + zoom * (near[0].pt.x + border_cam_x_dx);
					float mapped_y = hwin_y + zoom * (*near[0].pt.y * range_y + border_cam_y_dy);

					// determine which points are close to each other
					for (int i = 1; i < mouse_inspect_size; i++) {
						float mx = hwin_x + side_bar_w + zoom * (mouse_inspect[i].pt.x + border_cam_x_dx);
						float my = hwin_y + zoom * (*mouse_inspect[i].pt.y * range_y + border_cam_y_dy);

						float dx = mapped_x - mx;
						float dy = mapped_y - my;

						float dist = dx * dx + dy * dy;
						if (dist < 9) {
							near[near_size] = mouse_inspect[i];
							near_size++;
						}
					}

					inspect_point to_add[5];
					int add_size = 0;
					remove_pt to_remove[5];
					int remove_size = 0;

					// determine if the point needs to be removed or added
					for (int i = 0; i < near_size; i++) {
						double* y = near[i].pt.y;

						std::vector<double*>::iterator it = std::find(in_box.begin(), in_box.end(), y);

						if (it != in_box.end()) {
							in_box.erase(it);

							bool brk = false;

							for (int j = 0; j < inspect_boxes.size(); j++) {
								for (int k = 0; k < inspect_boxes[j].size; k++) {
									if (inspect_boxes[j].contents[k].y == y) {
										to_remove[remove_size] = {j, k};
										remove_size++;

										brk = true;
										break;
									}
								}

								if (brk) break;
							}
						} else {
							to_add[add_size] = near[i];
							add_size++;
						}
					}

					// determine which boxes already exist
					int boxes[5];
					int box_count = 0;

					for (int i = 0; i < remove_size; i++) {
						bool in = false;
						for (int j = 0; j < box_count; j++) {
							if (boxes[j] == to_remove[i].box_idx) {
								in = true;
								break;
							}
						}

						if (!in) {
							boxes[box_count] = to_remove[i].box_idx;
							box_count++;
						}
					}

					// sort
					remove_pt sorted_to_remove[5];
					int sort_remove_size = 0;

					for (int i = 0; i < remove_size; i++) {
						int j = 0;
						while (j < sort_remove_size && to_remove[i].pt_idx < sorted_to_remove[j].pt_idx) {
							j++;
						}

						int end = std::min(sort_remove_size, 4);
						for (int k = end; k > j; k--) {
							sorted_to_remove[k] = sorted_to_remove[k - 1];
						}

						sorted_to_remove[j] = to_remove[i];
						sort_remove_size++;
					}

					int remove_box[5];
					int remove_box_count = 0;

					// remove the points from their boxes
					for (int i = 0; i < sort_remove_size; i++) {
						inspect_box* box = &inspect_boxes[sorted_to_remove[i].box_idx];

						SDL_DestroyTexture(box->textures[sorted_to_remove[i].pt_idx]);
						box->textures[sorted_to_remove[i].pt_idx] = nullptr;

						for (int j = sorted_to_remove[i].pt_idx; j < box->size; j++) {
							box->contents[j] = box->contents[j + 1];
						}

						--box->size;

						if (box->size == 0) {
							for (int j = 0; j < box_count; j++) {
								if (sorted_to_remove[i].box_idx == boxes[j]) {

									for (int k = j; k < box_count; k++) {
										boxes[k] = boxes[k + 1];
									}

									box_count--;

									// negate boxes that no longer have any data
									remove_box[remove_box_count] = sorted_to_remove[i].box_idx;
									remove_box_count++;

									break;
								}
							}
						}
					}

					// sort
					inspect_point sorted[5] = {};
					int sorted_size = 0;

					for (int i = 0; i < add_size; i++) {
						int j = 0;
						while (j < sorted_size && (to_add[i].pt.x > sorted[j].pt.x || (to_add[i].pt.x == sorted[j].pt.x && to_add[i].pt.type > sorted[j].pt.type))) {
							j++;
						}

						int end = std::min(sorted_size, 4);
						for (int k = end; k > j; k--) {
							sorted[k] = sorted[k - 1];
						}

						sorted[j] = to_add[i];
						sorted_size++;
					}

					// add points to a new / existing box
					bool box_exists = false;
					inspect_box* box = {};

					if (box_count > 0) {
						box_exists = true;
						box = &inspect_boxes[boxes[0]];
					}

					for (int i = 0; i < add_size; i++) {
						if (!box_exists) {
							inspect_boxes.push_back(inspect_box{sorted[i].pt.x, sorted[i].pt.y, 0, 0, {}, {}, {}, {}, 0});

							box = &inspect_boxes.back();

							box_exists = true;
						}

						if (box->size < 12) {
							box->contents[box->size] = sorted[i].pt;
							++box->size;
							in_box.push_back(sorted[i].pt.y);
						}
					}

					// sort
					std::vector<int> sorted_ = {};

					for (int i = 0; i < remove_box_count; i++) {
						sorted_.push_back(remove_box[i]);
					}

					std::sort(sorted_.begin(), sorted_.end(), std::greater<int>());

					// remove boxes from the list that no longer exist
					for (int i = 0; i < sorted_.size(); i++) {
						inspect_boxes.erase(inspect_boxes.begin() + sorted_[i]);
						for (int j = 0; j < box_count; j++) {
							// addjust box indexes
							if (sorted_[i] < boxes[j]) {
								boxes[j] -= 1;
							}
						}
					}

					// (re)setup boxes
					for (int i = 0; i < box_count; i++) {
						setup_inspect_box(renderer, last, &inspect_boxes[boxes[i]], arial13, white, red_pt, blue_pt);
					}

					if (add_size > 0) {
						setup_inspect_box(renderer, last, &inspect_boxes.back(), arial13, white, red_pt, blue_pt);
					}
				}
			} else {
				box_added = false;
			}

			// sort points near the cursor by x position
			inspect_point sorted[5] = {};
			int sorted_size = 0;

			for (int i = 0; i < mouse_inspect_size; i++) {
				int j = 0;
				while (j < sorted_size && (mouse_inspect[i].pt.x > sorted[j].pt.x || (mouse_inspect[i].pt.x == sorted[j].pt.x && mouse_inspect[i].pt.type > sorted[j].pt.type))) {
					j++;
				}

				int end = std::min(sorted_size, 4);
				for (int k = end; k > j; k--) {
					sorted[k] = sorted[k - 1];
				}

				sorted[j] = mouse_inspect[i];
				sorted_size++;
			}

			// destroy old textures
			for (int i = 0; i < mouse_inspect_box.size; i++) {
				std::vector<double*>::iterator it = std::find(in_box.begin(), in_box.end(), mouse_inspect_box.contents[i].y);

				if (it == in_box.end()) {
					SDL_DestroyTexture(mouse_inspect_box.textures[i]);
					mouse_inspect_box.textures[i] = nullptr;
				}
			}

			// recreate inspect box
			mouse_inspect_box = {(float)mouse_x, (float)mouse_y, 0, 0, {}, {}, {}, {}, 0};

			// add contents
			int k = 0;
			for (int i = 0; i < mouse_inspect_size; i++) {
				std::vector<double*>::iterator it = std::find(in_box.begin(), in_box.end(), sorted[i].pt.y);

				if (it == in_box.end()) {
					mouse_inspect_box.contents[i - k] = sorted[i].pt;
					mouse_inspect_box.size++;
				} else {
					k++;
				}
			}

			// setup
			if (mouse_inspect_box.size == 0) {
				mouse_inspect_size = 0;
			} else {
				setup_mouse_inspect_box(renderer, &mouse_inspect_box, arial13, white, red_pt, blue_pt);
			}

			// render inspect boxes
			if (inspect_boxes.size() > 0) {
				for (auto& b : inspect_boxes) {
					render_inspect_box(renderer, side_bar_w, spacing, &b, box_tl, box_tr, box_br, box_bl, box_lside, box_rside, white_indicator, red_indicator, blue_indicator);
				}
			}

			if (mouse_inspect_box.size > 0) {
				render_mouse_box(renderer, &mouse_inspect_box, box_tl, box_tr, box_br, box_bl, box_lside, box_rside, white_indicator, red_indicator, blue_indicator);
			}
		} else if (inspect_boxes.size() > 0) {
			for (auto& box : inspect_boxes) {
				render_inspect_box(renderer, side_bar_w, spacing, &box, box_tl, box_tr, box_br, box_bl, box_lside, box_rside, white_indicator, red_indicator, blue_indicator);
			}

			if (!lmb) {
				box_added = false;
			}
		} else {
			if (!lmb) {
				box_added = false;
			}
		}

		// render settings button
		SDL_SetRenderDrawColor(renderer, 158, 158, 158, 255);
		SDL_RenderFillRect(renderer, &side_bar);

		if (over_settings) {
			SDL_RenderTexture(renderer, settings_button.icon_select, nullptr, &settings_button.rect);
		} else {
			SDL_RenderTexture(renderer, settings_button.icon, nullptr, &settings_button.rect);
		}

		// ---------------------------------------------------------------------------------- RENDER SETTINGS MENU ---------------------------------------------------------------------------------- 

		if (display_settings) {
			// main body
			SDL_SetRenderDrawColor(renderer, 158, 158, 158, 255);
			SDL_RenderFillRect(renderer, &open_side_bar);
			SDL_SetRenderDrawColor(renderer, 130, 130, 130, 255);
			SDL_RenderFillRect(renderer, &open_bar_border);

			// toggles
			SDL_RenderTexture(renderer, gridlines_label.text, nullptr, &gridlines_label.rect);
			if (*gridlines_toggle.state) {
				SDL_RenderTexture(renderer, toggle_on, nullptr, &gridlines_toggle.rect);
			} else {
				SDL_RenderTexture(renderer, toggle_off, nullptr, &gridlines_toggle.rect);
			}

			SDL_RenderTexture(renderer, percents_label.text, nullptr, &percents_label.rect);
			if (*percents_toggle.state) {
				SDL_RenderTexture(renderer, toggle_on, nullptr, &percents_toggle.rect);
			} else {
				SDL_RenderTexture(renderer, toggle_off, nullptr, &percents_toggle.rect);
			}

			SDL_RenderTexture(renderer, equation_label.text, nullptr, &equation_label.rect);
			if (*equation_toggle.state) {
				SDL_RenderTexture(renderer, toggle_on, nullptr, &equation_toggle.rect);
			} else {
				SDL_RenderTexture(renderer, toggle_off, nullptr, &equation_toggle.rect);
			}

			SDL_RenderTexture(renderer, petal_label.text, nullptr, &petal_label.rect);
			if (*petals_toggle.state) {
				SDL_RenderTexture(renderer, toggle_on, nullptr, &petals_toggle.rect);
			} else {
				SDL_RenderTexture(renderer, toggle_off, nullptr, &petals_toggle.rect);
			}

			// dividers
			SDL_SetRenderDrawColor(renderer, 144, 144, 144, 255);
			SDL_RenderFillRect(renderer, &divider_1);
			SDL_RenderFillRect(renderer, &divider_2);
			SDL_RenderFillRect(renderer, &divider_3);
			SDL_RenderFillRect(renderer, &divider_4);

			// textbox
			SDL_RenderTexture(renderer, textbox, nullptr, &textbox_rect);

			// input
			SDL_DestroyTexture(input_text);
			input_text = create_text_texture(renderer, current_input, arial13, black);

			SDL_GetTextureSize(input_text, &w, &h);
			input_rect.w = w;
			input_rect.h = h;

			while (w > 222) {
				current_input.pop_back();
				caret_index--;

				SDL_DestroyTexture(input_text);
				input_text = create_text_texture(renderer, current_input, arial13, black);

				SDL_GetTextureSize(input_text, &w, &h);
				input_rect.w = w;
				input_rect.h = h;
			}

			if (caret_index == current_input.size() || caret_clock < caret_begin || !input_active) {
				SDL_RenderTexture(renderer, input_text, nullptr, &input_rect);

				if (input_active) {
					caret_clock++;

					if (caret_clock > caret_begin) {
						caret_rect.x = side_bar_width + 15 + w;

						SDL_SetRenderDrawColor(renderer, 30, 30, 30, 255);
						SDL_RenderFillRect(renderer, &caret_rect);

						caret_clock %= (int)fps_limit;
					}
				}
			} else {
				SDL_RenderTexture(renderer, input_text, nullptr, &input_rect);

				SDL_DestroyTexture(input_text);
				input_text = create_text_texture(renderer, current_input.substr(0, caret_index), arial13, black);

				SDL_GetTextureSize(input_text, &w, &h);

				if (input_active) {
					caret_clock++;

					if (caret_clock > caret_begin) {
						caret_rect.x = side_bar_width + 15 + w;

						SDL_SetRenderDrawColor(renderer, 30, 30, 30, 255);
						SDL_RenderFillRect(renderer, &caret_rect);

						caret_clock %= (int)fps_limit;
					}
				}
			}

			// dropdown menu
			if (dropdown_bool) {
				SDL_RenderTexture(renderer, dropdown_open, nullptr, &dropdown_rect);


				extension_rect.y = dropdown_rect.y;
				for (int i = 0; i < dropdown_displays.size() - 2; i++) {
					extension_rect.y += increment;
					SDL_RenderTexture(renderer, dropdown_extension, nullptr, &extension_rect);

				}
				SDL_FRect var_rect = {dropdown_display_rects[0].x, dropdown_display_rects[0].y, dropdown_display_rects[selected_var].w, dropdown_display_rects[selected_var].h};

				SDL_RenderTexture(renderer, dropdown_displays[selected_var], nullptr, &var_rect);

				int j = 0;
				int i;
				for (i = 1; i < dropdown_displays.size() - 1; i++, j++) {
					if (selected_var == j) {
						j++;
					}

					var_rect = {dropdown_display_rects[i].x, dropdown_display_rects[i].y, dropdown_display_rects[j].w, dropdown_display_rects[j].h};

					SDL_RenderTexture(renderer, dropdown_displays[j], nullptr, &var_rect);
				}

				if (display_equation) {
					extension_rect.y += increment;
					SDL_RenderTexture(renderer, dropdown_extension, nullptr, &extension_rect);

					if (selected_var == j) {
						j++;
					}

					var_rect = {dropdown_display_rects[i].x, dropdown_display_rects[i].y, dropdown_display_rects[j].w, dropdown_display_rects[j].h};

					SDL_RenderTexture(renderer, dropdown_displays[j], nullptr, &var_rect);
				}

			} else {
				SDL_RenderTexture(renderer, dropdown_closed, nullptr, &dropdown_rect);
				SDL_FRect var_rect = {dropdown_display_rects[0].x, dropdown_display_rects[0].y, dropdown_display_rects[selected_var].w, dropdown_display_rects[selected_var].h};
				SDL_RenderTexture(renderer, dropdown_displays[selected_var], nullptr, &var_rect);
			}

			if (assign_error_clock > 0) {
				SDL_RenderTexture(renderer, assignment_error, nullptr, &assign_error_rect);
				assign_error_clock--;
			}

			// recalculate button 
			if (over_recalculate) {
				SDL_RenderTexture(renderer, recalculate_button.icon_select, nullptr, &recalculate_button.rect);
			} else {
				SDL_RenderTexture(renderer, recalculate_button.icon, nullptr, &recalculate_button.rect);
			}

			//clear button 
			if (inspect_boxes.size() > 0) {
				if (over_clear) {
					SDL_RenderTexture(renderer, clear_button.icon_select, nullptr, &clear_button.rect);
				} else {
					SDL_RenderTexture(renderer, clear_button.icon, nullptr, &clear_button.rect);
				}
			}

			//info 

			float dx = mouse_x - info_icon_x;
			float dy = mouse_y - info_icon_y;

			if (dx * dx + dy * dy < info_icon_r2 || info_clicked) {
				SDL_RenderTexture(renderer, info_select, nullptr, &info_icon_rect);

				if (lmb) {
					if (!info_clicked) {
						display_info = !display_info;

						ip_scroll_y = 0;

						info_clicked = true;
					}
				} else {
					info_clicked = false;
				}

			} else {
				SDL_RenderTexture(renderer, info_icon, nullptr, &info_icon_rect);
			}

			if (display_info) {
				SDL_RenderTexture(renderer, info_box, nullptr, &info_box_rect);
				info_panel_rect.y = ip_scroll_y;
				SDL_RenderTexture(renderer, info_panel, &info_panel_rect, &info_prj_rect);
				SDL_RenderTexture(renderer, scroll_bar, nullptr, &scroll_bar_rect);
			}

		} else {
			SDL_SetRenderDrawColor(renderer, 130, 130, 130, 255);
			SDL_RenderFillRect(renderer, &bar_border);
		}

		// update window
		SDL_RenderPresent(renderer);

		// frame delay
		frame_time = SDL_GetPerformanceCounter() * inv_freq - frame_start;

		if (frame_time < delay) {
			SDL_Delay(delay - frame_time);
		}
	}

	frame_barrier.arrive_and_wait();

	// destroy tetures to prevent memory leakage
	SDL_StopTextInput(window);
	SDL_DestroyTexture(black_point);
	SDL_DestroyTexture(blue_point);
	SDL_DestroyTexture(red_point);
	SDL_DestroyTexture(settings_button.icon);
	SDL_DestroyTexture(settings_button.icon_select);
	SDL_DestroyTexture(recalculate_button.icon);
	SDL_DestroyTexture(recalculate_button.icon_select);
	SDL_DestroyTexture(clear_button.icon);
	SDL_DestroyTexture(clear_button.icon_select);
	SDL_DestroyTexture(dropdown_closed);
	SDL_DestroyTexture(dropdown_open);
	SDL_DestroyTexture(dropdown_extension);
	SDL_DestroyTexture(assignment_error);
	SDL_DestroyTexture(box_tl);
	SDL_DestroyTexture(box_tr);
	SDL_DestroyTexture(box_br);
	SDL_DestroyTexture(box_bl);
	SDL_DestroyTexture(box_lside);
	SDL_DestroyTexture(box_rside);
	SDL_DestroyTexture(white_indicator);
	SDL_DestroyTexture(red_indicator);
	SDL_DestroyTexture(blue_indicator);
	for (auto& dsp : dropdown_displays) {
		SDL_DestroyTexture(dsp);
	}
	for (auto& lbl : vertical_labels) {
		SDL_DestroyTexture(lbl);
	}
	for (auto& lbl : horizontal_labels) {
		SDL_DestroyTexture(lbl);
	}
	if (mouse_inspect_box.size > 0) {
		for (auto& tx : mouse_inspect_box.textures) {
			SDL_DestroyTexture(tx);
		}
	}
	if (inspect_boxes.size() > 0) {
		for (int i = 0; i < inspect_boxes.size(); i++) {
			for (auto& tx : inspect_boxes[i].textures) {
				SDL_DestroyTexture(tx);
			}
		}
	}
	SDL_DestroyTexture(textbox);
	SDL_DestroyTexture(input_text);
	SDL_DestroyTexture(toggle_on);
	SDL_DestroyTexture(toggle_off);
	SDL_DestroyTexture(info_panel);
	SDL_DestroyTexture(info_icon);
	SDL_DestroyTexture(info_select);
	SDL_DestroyTexture(info_box);
	SDL_DestroyTexture(average_display);
	SDL_DestroyTexture(average_trials_dislay);
	SDL_DestroyTexture(scroll_bar);

	TTF_CloseFont(arial13);
	TTF_CloseFont(arial10);

	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();
	TTF_Quit();
	return 0;
}