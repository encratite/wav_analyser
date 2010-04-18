#include <iostream>
#include <vector>
#include <cmath>

#include <fftw3.h>

#include <ail/file.hpp>
#include <ail/string.hpp>
#include <ail/bmp.hpp>
#include <ail/array.hpp>

struct sample_pair
{
	long minimum, maximum;

	sample_pair():
		minimum(0),
		maximum(0)
	{
	}

	sample_pair(long minimum, long maximum):
		minimum(minimum),
		maximum(maximum)
	{
	}
};

typedef double fft_type;

namespace
{
	sample_pair
		left_pair,
		right_pair;

	fft_type
		* left_table,
		* right_table;

	fftw_complex * fft_output;

	fftw_plan
		left_plan,
		right_plan;

	std::size_t pair_counter = 0;

	std::size_t const border_height = 4;
}

long get_sample(char * data, std::size_t offset)
{
	offset *= 2;
	long output = data[offset] | (data[offset + 1] << 8);
	return output;
}

void wave_form_check(sample_pair & current_pair, long sample)
{
	if(sample > current_pair.maximum)
		current_pair.maximum = sample;
	else if(sample < current_pair.minimum)
		current_pair.minimum = sample;
}

std::string get_memory(std::size_t size)
{
	fft_type output = static_cast<fft_type>(size);

	std::string const units[] =
	{
		"B",
		"KiB",
		"MiB",
		"GiB"
	};

	fft_type const power = 1024.0f;

	std::size_t unit_offset = 0;
	for(; output > power && unit_offset < ail::countof(units) - 1; unit_offset++, output /= power);

	return ail::number_to_string<fft_type>(output) + " " + units[unit_offset];
}

void fix_coordinate(std::size_t & y, std::size_t initial_y, std::size_t height)
{
	std::size_t minimum = initial_y - border_height;
	std::size_t maximum = initial_y + height + border_height;
	if(y < minimum)
		y = minimum;
	else if(y >= maximum)
		y = maximum - 1;
}

void process_wave_form(ail::bmp & bmp, std::size_t x, std::size_t y, std::size_t height, sample_pair const & pair)
{
	enum method_type
	{
		method_one_colour,
		method_global_gradient,
		method_individual_gradient,
		method_individual_exponential_gradient
	};
	long const outer_colour = 0x000000;
	long const inner_colour1 = 0xff;
	long const inner_colour2 = 0x00;

	//method_type const method = method_individual_exponential_gradient;
	method_type const method = method_one_colour;

	std::size_t amplitude = height / 2;
	std::size_t const maximum = 0x10000 >> 1;
	double factor = static_cast<double>(amplitude) / maximum;
	std::size_t middle_y = y + amplitude;
	std::size_t upper_y = static_cast<std::size_t>(static_cast<double>(middle_y) - pair.maximum * factor);
	std::size_t lower_y = static_cast<std::size_t>(static_cast<double>(middle_y) - pair.minimum * factor);
	//std::cout << "Pre fix: (" << upper_y << ", " << lower_y << ")" << std::endl;
	fix_coordinate(upper_y, y, height);
	fix_coordinate(lower_y, y, height);
	//std::cout << "(" << upper_y << ", " << lower_y << ")" << std::endl;
	for(std::size_t i = y; i < upper_y; i++)
		bmp.set_pixel(x, i, outer_colour);
	for(std::size_t i = upper_y; i <= lower_y; i++)
	{
		double progress;
		long colour;
		switch(method)
		{
			case method_one_colour:
				colour = 0xffffff;
				break;

			case method_global_gradient:
				progress = std::fabs((static_cast<double>(i) - static_cast<double>(y + amplitude)) / amplitude);
				break;

			case method_individual_gradient:
			case method_individual_exponential_gradient:
			{

				double divisor;
				if(i < middle_y)
					divisor = static_cast<double>(middle_y - upper_y);
				else
					divisor = static_cast<double>(lower_y - middle_y);

				if(divisor == 0)
					progress = 0.0;
				else
					progress = std::fabs((static_cast<double>(i) - static_cast<double>(middle_y)) / divisor);

				if(progress < 0.0)
					progress = 0.0;
				else if(progress > 1.0)
					progress = 1.0;

				if(method == method_individual_exponential_gradient)
					progress = std::pow(progress, 8.0);
				break;
			}
		}
		
		if(method != method_one_colour)
		{
			long pixel = static_cast<long>(progress * inner_colour2 + (1.0 - progress) * inner_colour1);
			colour = (pixel << 16) | (pixel << 8) | pixel;
		}

		bmp.set_pixel(x, i, colour);
	}
	for(std::size_t i = lower_y + 1; i < y + height; i++)
		bmp.set_pixel(x, i, outer_colour);
}

void create_spectrogram(ail::bmp & bmp, std::size_t x, std::size_t y, std::size_t height, fftw_plan & plan, std::size_t channel_size)
{
	fftw_execute(plan);

	double const spectrum_usage = 0.2;

	std::size_t bands_per_pixel = static_cast<std::size_t>(channel_size / height / 2 * spectrum_usage);
	double sum = 0.0;
	std::size_t counter = 0;
	std::size_t offset = 0;
	for(long i = 0, end = static_cast<long>(channel_size / 2); offset < height && i < end; i++)
	{
		fftw_complex const & sample = fft_output[i];

		double const & real = sample[0];
		double const & imaginary = sample[1];

		double value = std::sqrt(real * real + imaginary * imaginary);

		sum += value;
		counter++;
		if(counter == bands_per_pixel)
		{
			double arithmetic_mean = (0xff * (sum / bands_per_pixel)) / 0xffff * 0.02676;
			//double arithmetic_mean = (0xff * (sum / bands_per_pixel)) / (0xffff >> 1);
			long pixel = static_cast<long>(arithmetic_mean);
			if(pixel > 0xff)
				pixel = 0xff;
			else if(pixel < 0)
				pixel = 0;
			long colour = (pixel << 16) | (pixel << 8) | pixel;
			bmp.set_pixel(x, y + height - offset, colour);
			sum = 0.0f;
			counter = 0;
			offset++;
		}
	}
}

void process_data(char * data, std::size_t & x, std::size_t samples_per_pixel, std::size_t channel_height, std::size_t size, ail::bmp & bmp)
{
	std::size_t samples = size / 2;
	
	for(std::size_t i = 0; i < samples; i++)
	{
		long sample = get_sample(data, i);
		fft_type table_sample = static_cast<fft_type>(sample);

		if(i % 2 == 0)
		{
			wave_form_check(left_pair, sample);
			left_table[pair_counter] = table_sample;
		}
		else
		{
			wave_form_check(right_pair, sample);
			right_table[pair_counter] = table_sample;
			pair_counter++;
		}

		if(pair_counter == samples_per_pixel)
		{
			pair_counter = 0;

			process_wave_form(bmp, x, border_height, channel_height, left_pair);
			process_wave_form(bmp, x, 3 * border_height + channel_height, channel_height, right_pair);

			create_spectrogram(bmp, x, 5 * border_height + 2 * channel_height, channel_height, left_plan, samples_per_pixel);
			create_spectrogram(bmp, x, 7 * border_height + 3 * channel_height, channel_height, right_plan, samples_per_pixel);

			left_pair = sample_pair();
			right_pair = sample_pair();
			x++;
		}
	}
}

int main(int argc, char ** argv)
{
	if(argc != 5)
	{
		std::cout << argv[0] << " <input WAV> <output BMP> <width> <height>" << std::endl;
		return 1;
	}
	std::string wav_path = argv[1];
	std::string bmp_path = argv[2];
	std::size_t width = ail::string_to_number<std::size_t>(argv[3]);
	std::size_t height = ail::string_to_number<std::size_t>(argv[4]);
	std::string wav_data;
	ail::file wav_file;
	if(!wav_file.open(wav_path, ail::file::open_mode_read))
	{
		std::cout << "Failed to read input file" << std::endl;
		return 1;
	}
	std::size_t const header_size = 44;
	std::size_t data_size = wav_file.get_size() - header_size;
	wav_file.set_file_pointer(header_size);
	std::size_t const chunk_size = 8 * 1024 * 1024;

	std::cout << "Allocating WAV buffer (" << get_memory(chunk_size) << ")" << std::endl;

	char * wav_buffer = new char[chunk_size];
	std::size_t samples_per_channel = data_size / 4;

	std::size_t samples_per_pixel = samples_per_channel / width;

	std::size_t fft_channel_size = sizeof(fft_type) * samples_per_pixel;
	std::size_t output_table_size = sizeof(fftw_complex) * samples_per_pixel;

	std::cout << "Samples per pixel: " << samples_per_pixel << std::endl;

	std::cout << "Allocating FFT tables (" << get_memory(2 * fft_channel_size + output_table_size) << ")" << std::endl;

	left_table = reinterpret_cast<fft_type *>(fftw_malloc(fft_channel_size));
	right_table = reinterpret_cast<fft_type *>(fftw_malloc(fft_channel_size));
	fft_output = reinterpret_cast<fftw_complex *>(fftw_malloc(output_table_size));

	/*
	left_plan = fftw_plan_r2r_1d(static_cast<int>(samples_per_pixel), left_table, fft_output, FFTW_DHT, FFTW_ESTIMATE);
	right_plan = fftw_plan_r2r_1d(static_cast<int>(samples_per_pixel), right_table, fft_output, FFTW_DHT, FFTW_ESTIMATE);
	*/

	left_plan = fftw_plan_dft_r2c_1d(static_cast<int>(samples_per_pixel), left_table, fft_output, FFTW_ESTIMATE);
	right_plan = fftw_plan_dft_r2c_1d(static_cast<int>(samples_per_pixel), right_table, fft_output, FFTW_ESTIMATE);

	std::size_t channel_size = samples_per_channel / samples_per_pixel;

	ail::bmp bmp;
	std::size_t bmp_size = bmp.process_image_size(width, height);
	std::cout << "Allocating BMP memory (" << get_memory(bmp_size) << ")" << std::endl;
	bmp.initialise(width, height);

	std::cout << "Processing WAV data" << std::endl;

	std::size_t channel_count = 4;
	std::size_t x = 0;
	std::size_t channel_height = (height - 2 * channel_count * border_height) / channel_count;

	for(std::size_t offset = 0; offset < data_size; offset += chunk_size)
	{
		std::cout << "Processing chunk " << (offset / chunk_size) << std::endl;
		std::size_t current_chunk_size;
		std::size_t remaining_size = data_size - offset;
		if(remaining_size < chunk_size)
			current_chunk_size = remaining_size;
		else
			current_chunk_size = chunk_size;
		wav_file.read(wav_buffer, current_chunk_size);
		process_data(wav_buffer, x, samples_per_pixel, channel_height, current_chunk_size, bmp);
	}

	std::cout << "Deallocating WAV buffer" << std::endl;
	delete wav_buffer;

	std::cout << "Deallocating FFT tables" << std::endl;
	fftw_free(left_table);
	fftw_free(right_table);
	fftw_free(fft_output);

	fftw_destroy_plan(left_plan);
	fftw_destroy_plan(right_plan);

	std::cout << "Writing BMP data to " << bmp_path << std::endl;
	bmp.write(bmp_path);

	return 0;
}
