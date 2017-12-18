
#include <stdio.h>
#include <string.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <algorithm>
#include <random>
#include <functional>
#include <vector>
#include <memory>
#include <thread>
#include <numeric>
#include <chrono>
#include <map>

#include "Reader.h"
#include "Sparsity.h"

// due to portability issues
#ifdef _MSC_VER
#define fseeko _fseeki64
#define ftello _ftelli64
#endif // _MSC_VER

#ifndef Float
#	define Float double
#endif

Vocabulary V;
Sparsity sparsity;

std::string unk = "unk";

// typedef std::atomic<_FLOAT> Float;

Float* M, *gradsq;

unsigned int thread = 1;

std::string corpus = "";

unsigned int window = 5;
unsigned int negative = 1;
Float eta = 0.1;

std::vector<Float> eye_m; //!< identity matrix, in sparse representation

std::string load_model = "", vocab_filename = "";
std::string output_filename = "matrices.me";
std::string sparsity_structure = "[(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]";

unsigned int epoch = 1;
Float total_cost;
size_t total_processed_words;

//! this signals the halt
bool terminated = false;

//! this is substitute for `basename`, due to portability issues.
/*!
	see http://stackoverflow.com/a/5804935/3583290
*/
const char* gnu_basename(const char *path)
{
	const char *base = std::max(strrchr(path, '/'), strrchr(path, '\\'));
	return base ? base + 1 : path;
}

size_t ReadModel(std::string filename)
{
	size_t total_param_number = 0;
	std::string str;
	std::ifstream fin(filename);
	M = nullptr;
	V.clear();
	if (fin)
	{
		std::getline(fin, str);
		sparsity.InitKernels(Sparsity::ParseLine(str));

		size_t rows = 0;
		while (std::getline(fin, str))
			++rows;
		
		fin.close();
		fin.open(filename);
		std::getline(fin, str);

		const size_t paramNumber = sparsity.GetParameterNumber();
		total_param_number = rows * paramNumber;
		M = new (std::nothrow) Float[total_param_number];

		if (M)
		{
			for (size_t i = 0; i < rows; ++i)
			{
				if (fin >> str)
				{
					V.GetIndex(str);
					for (size_t j = 0; j < paramNumber; ++j)
					{
						fin >> M[i * paramNumber + j];
					}
				}
				else
					return i * paramNumber;
			}
		}
	}
	return total_param_number;
}

bool WriteModel(const std::string& filename)
{
	std::ofstream ofs(filename);
	if (ofs.good())
	{
		std::map<size_t, Sparsity::Position> ordered_positions;
		for (const auto& pos : sparsity.GetPositions())
			ordered_positions[pos.second] = pos.first;

		const size_t paramNumber = sparsity.GetParameterNumber();
		ofs << '[';
		for (const auto& pos : ordered_positions)
			ofs << '(' << pos.second.first << ", " << pos.second.second << "), ";
		ofs << ']' << std::endl;
		
		for (size_t i = 0; i < V.size(); ++i)
		{
			ofs << V.GetWord(i);
			for (size_t j = 0; j < paramNumber; ++j)
				ofs << ' ' << M[i * paramNumber + j];
			ofs << std::endl;
		}
		return ofs.good();
	}
	else
		return false;
}

class vector : public std::vector<Float>
{
	typedef std::vector<Float> Parent;
public:
	vector(size_t size) : Parent(size){}
	template<class Type>
	vector& operator= (Type x)
	{
		const auto end = data() + size();
		for (auto i = data(); i < end; ++i)
			*i = x;

		return *this;
	}
	vector& operator= (const Parent& right)
	{
		*(Parent*)(this) = right;
		return *this;
	}
	operator Parent ()
	{
		return *this;
	}
	operator const Parent () const
	{
		return *this;
	}
};

bool ThreadProcedure(size_t id)
{
	const auto paramNumber = sparsity.GetParameterNumber();

	vector buffer(paramNumber), left(paramNumber), right(paramNumber), backprop(paramNumber);
	Float batch_sum, negative_batch_sum, scalar;
	Float cost;
	std::vector<size_t> negative_samples;
	std::vector<Float> negative_values;
	size_t offset, index;
	size_t positive_sample;
	Float tmp;

	std::default_random_engine generator(123u << id);
	std::uniform_int_distribution<size_t> distribution(0, V.size()-1);
	auto negative_sampler = std::bind(distribution, generator);

	//due to \r\n file endings on windows, this should be opened as binary
	FILE* cfin = fopen(corpus.c_str(), "rb");
	if (cfin == nullptr)
		return false;

	if (fseeko(cfin, 0, SEEK_END) != 0)
		return false;

	const auto file_size = ftello(cfin);
	const auto end = 1 + (size_t)(file_size * (id + 1)) / thread;

	if (fseeko(cfin, (file_size * id) / thread, SEEK_SET) != 0)
		return false;

	const auto begin = ftello(cfin);

	Reader reader(cfin);
	if (id > 0) //swallow first word
		reader.ReadNext();

	WindowsReader ir(reader, V, window, window);

	while (ir.IsGood() && ir.GetPosition() + begin < end && !terminated)
	{
		positive_sample = ir.GetContext()[ir.GetWord()];
		// calculate left matrix
		if (ir.GetWord() == 0)
			left = eye_m; // left context is empty
		else
		{
			offset = ir.GetContext()[0] * paramNumber;
			left.assign(M + offset, M + offset + paramNumber);
			for (size_t l = 1; l < ir.GetWord(); ++l)
			{
				buffer = 0;
				sparsity.DotProduct(left.data(), M + ir.GetContext()[l] * paramNumber, buffer.data());
				std::swap(buffer, left);
			}
		}
		// calculate right matrix
		if (ir.GetWord() + 1 == ir.GetContext().size())
			right = eye_m; // right context is empty
		else
		{
			offset = ir.GetContext()[ir.GetWord() + 1] * paramNumber;
			right.assign(M + offset, M + offset + paramNumber);
			for (size_t r = ir.GetWord() + 2; r < ir.GetContext().size(); ++r)
			{
				buffer = 0;
				sparsity.DotProduct(right.data(), M + ir.GetContext()[r] * paramNumber, buffer.data());
				std::swap(buffer, right);
			}
		}

		offset = positive_sample * paramNumber;
		tmp = sparsity.Sandwitch(left.data(), M + offset, right.data());
		batch_sum = exp(tmp);
		cost = tmp;

		backprop = 0;
		sparsity.BackPropagate(left.data(), right.data(), backprop.data());

		negative_batch_sum = 0;
		negative_samples.clear();
		negative_values.clear();
		for (size_t n = 0; n < negative; ++n)
		{
			size_t next_negative_sample;
			do 
			{
				next_negative_sample = negative_sampler();
			} while (std::find(negative_samples.begin(), negative_samples.end(), next_negative_sample) != negative_samples.end() || next_negative_sample == positive_sample);
			negative_samples.push_back(next_negative_sample);
			negative_values.push_back(exp(sparsity.Sandwitch(left.data(), M + negative_samples.back() * paramNumber, right.data())));
			batch_sum += negative_values.back();
			negative_batch_sum += negative_values.back();
		}

		cost -= log(batch_sum);
		
		scalar = negative_batch_sum / batch_sum;
		if (std::fpclassify(scalar) == FP_NORMAL)
		{
            total_cost -= cost;

            for (size_t i = 0; i < paramNumber; ++i)
            {
                index = offset + i;
                tmp = scalar * backprop[i];
                M[index] += tmp / sqrt(gradsq[index]);
                gradsq[index] += tmp*tmp;
            }
            for (size_t n = 0; n < negative; ++n)
            {
                scalar = negative_values[n] / batch_sum;
                offset = negative_samples[n] * paramNumber;
                for (size_t i = 0; i < paramNumber; ++i)
                {
                    index = offset + i;
                    tmp = scalar * backprop[i];
                    M[index] -= tmp / sqrt(gradsq[index]);
                    gradsq[index] += tmp*tmp;
                }
            }

            ++total_processed_words;
        }
        else
        {
			terminated = true;
			std::cerr << "Thread " << id << " encountered '" << scalar<< "' at file position " << ftello(cfin) << std::endl;			
			size_t i = 0;
			std::cerr << '[';
			for (; i < ir.GetWord(); ++i)
				std::cerr << ' ' << V.GetWord(ir.GetContext()[i], unk);
			std::cerr << "] " << V.GetWord(ir.GetContext()[ir.GetWord()], unk) << " [";
			for (i = ir.GetWord() + 1; i < ir.GetContext().size(); ++i)
				std::cerr << ' ' << V.GetWord(ir.GetContext()[i], unk);
			std::cerr << ']' << std::endl;
			fclose(cfin);
			return false;
		}

		ir.ReadItem();
	}

	fclose(cfin);

	return true;
}

bool InitializeModel()
{
	if (load_model.empty())
	{
		std::cerr << "Reading vocabulary from \"" << vocab_filename << "\" ... ";
		std::cerr.flush();
		{
			std::ifstream vocab_f(vocab_filename);
			V.Read(vocab_f);
		}
		std::cerr << "Done" << std::endl;

		std::cerr << "Initializing model ... ";
		std::cerr.flush();

		{// reading sparsity mask from file
			std::ifstream ifs(sparsity_structure);
			std::vector<std::vector<bool>> mask;
			if (ifs)
			{
				std::string line;
				while (std::getline(ifs, line))
				{
					mask.emplace_back();
					for (auto c : line)
					{
						switch (c)
						{
						case ' ':
						case '\t':
							mask.back().push_back(false);
							break;
						default:
							mask.back().push_back(true);
							break;
						}
					}
				}
				sparsity.InitKernels(mask);
			}else
			{ // parse as if it were a list of positions
				sparsity.InitKernels(Sparsity::ParseLine(sparsity_structure));
			}
			
		}

		const size_t totalParamNumber = V.size() * sparsity.GetParameterNumber();
		M = new (std::nothrow) Float[totalParamNumber];
		if (M == nullptr)
		{
			std::cerr << "Could not allocate memory for model parameters (";
			std::cerr << sizeof(Float) << " * " << V.size() << " * " << sparsity.GetParameterNumber() << " = " << sizeof(Float) * totalParamNumber << " bytes)!";
			std::cerr << std::endl;
			return false;
		}

		std::default_random_engine generator(1234u);
		std::normal_distribution<Float> distribution(pow(1.0/(sparsity.GetDim() * V.size()), 1.0 / window), 1.0 / window);
		auto random = std::bind(distribution, generator);

		for (size_t i = 0; i < totalParamNumber; ++i)
			M[i] = random();
		//if (!WriteModel(output_filename))
		//	return EXIT_FAILURE;
	}
	else
	{
		std::cerr << "Reading model from \"" << load_model << "\" ... ";
		std::cerr.flush();
		const auto size = ReadModel(load_model);
		if (M == nullptr)
		{
			std::cerr << "failed!" << std::endl;
			return false;
		}

		if (size != V.size() * sparsity.GetParameterNumber())
		{
			std::cerr << "Model size and other parameters did not match!" << std::endl;
			return false;
		}
	}
	//! check for UNK
	if (V.GetIndexInvalid(unk) == V.size())
	{
		std::cerr << "Unknown token \"" << unk << "\" not in the vocabulary" << std::endl;
		return false;
	}
	// todo eos, sos
	//for (auto item : std::vector<std::string>({ unk, "" }))
	//	if (V.find(item) == V.end())
	//	{
	//		std::cerr << "End-of-sentence or Unknown symbol not found!" << std::endl;
	//		return EXIT_FAILURE;
	//	}

	std::cerr << "with " << V.size() << "*" << sparsity.GetParameterNumber() << " parameters ... ";
	std::cerr.flush();

	eye_m.assign(sparsity.GetParameterNumber(), (Float)0);
	for (auto i : sparsity.GetOnes())
		eye_m[i] = (Float)1;

	{// initialize gradsq
		const size_t totalParamNumber = V.size() * sparsity.GetParameterNumber();
		gradsq = new (std::nothrow) Float[totalParamNumber];
		if (gradsq == nullptr)
		{
			std::cerr << "Unable to allocate " << sizeof(Float) << " * " << V.size() << " * " << sparsity.GetParameterNumber() <<
				" = " << (sizeof(Float) * totalParamNumber) << "bytes! " << std::endl;
			return false;
		}
		const Float c = 1.0 / (eta * eta);
		for (size_t i = 0; i < totalParamNumber; ++i)
			gradsq[i] = c;
	}

	std::cerr << "Done" << std::endl;

	if (!sparsity.IsAssociative())
		std::cerr << "WARNING: matrix algebra is not associative!" << std::endl;

	return true;
}

int main(int argc, char **argv)
{
	for (++argv, --argc; argc > 0; ++argv, --argc)
	{
		std::string arg(*argv);
		if ((arg == "-m" || arg == "--model") && argc > 1)
		{
			++argv; --argc;
			load_model = *argv;
		}
		else if ((arg == "-n" || arg == "--negative") && argc > 1)
		{
			++argv; --argc;
			negative = atoi(*argv);
		}
		else if ((arg == "-c" || arg == "--corpus") && argc > 1)
		{
			++argv; --argc;
			corpus = *argv;
		}
		else if ((arg == "-w" || arg == "--window") && argc > 1)
		{
			++argv; --argc;
			window = atoi(*argv);
		}
		else if ((arg == "-e" || arg == "--epoch") && argc > 1)
		{
			++argv; --argc;
			epoch = atoi(*argv);
		}
		else if ((arg == "-v" || arg == "--vocab") && argc > 1)
		{
			++argv; --argc;
			vocab_filename = *argv;
		}
		else if ((arg == "-o" || arg == "--output") && argc > 1)
		{
			++argv; --argc;
			output_filename = *argv;
		}
		else if ((arg == "-t" || arg == "--thread") && argc > 1)
		{
			++argv; --argc;
			thread = atoi(*argv);
		}
		else if ((arg == "-l" || arg == "--learning_rate") && argc > 1)
		{
			++argv; --argc;
			eta = (Float)atof(*argv);
		}
		else if ((arg == "-u" || arg == "--unk") && argc > 1)
		{
			++argv; --argc;
			unk = *argv;
		}
		else if ((arg == "-s" || arg == "--sparsity")&& argc > 1)
		{
			++argv; --argc;
			sparsity_structure = *argv;
		}
		else if (arg == "-h" || arg == "--help")
		{
            printf("                                      [1]\n");
            printf("                                      [1]\n");
            printf("                                      [1]\n");
			printf("        [       ] [ X     ] [       ]\n");
            printf("        [  the  ] [ X X   ] [ barks ]\n");
            printf("        [       ] [ X X X ] [       ]\n");
            printf("[1 1 1]                              \n");
            printf("Language Modeling with Matrix Embedding\n");
            printf("Author: Gabor Borbely - borbely@math.bme.hu\n\n");
			printf("Command line arguments:\n");
			printf("\t-m\t--model\t<string>\tinitial model file name, you can load a previous model, default \"%s\"\n", load_model.c_str());
			printf("\t-n\t--negative\t<uint>\tnumber of negative samples per learning examples, default %u\n", negative);
			printf("\t-c\t--corpus\t<string>\tthe input text file, empty is stdin, default \"%s\"\n", corpus.c_str());
			printf("\t-w\t--windows\t<uint>\twindows size of the context before and after the current word, default %u\n", window);
			printf("\t-e\t--epoch\t<uint>\tnumber of epochs, default %u\n", epoch);
			printf("\t-v\t--vocab\t<string>\tvocabulary file to initialize the matrices, if empty then one will be built for you, default \"%s\"\n", vocab_filename.c_str());
			printf("\t-o\t--output\t<string>\tname for the output model, default \"%s\"\n", output_filename.c_str());
			printf("\t-t\t--thread\t<uint>\tnumber of working threads, default %u\n", thread);
			printf("\t-l\t--learning_rate\t<float>\tinitial learning rate, default %g\n", eta);
			printf("\t-s\t--sparsity\t<str>\tsparsity structure, or filename of mask, default \"%s\"\n", sparsity_structure.c_str());
			printf("\t-u\t--unk\t<string>\tspecial unknown token, default \"%s\"\n", unk.c_str());
			return 0;
		}
		else
			std::cerr << "Unknown argument \"" << arg <<"\"!" << std::endl;
	}

	/************************************************************************/
	/* TODO check the values of the parameters                              */
	/************************************************************************/

	if (!InitializeModel())
		return 1;
	
	V.SetUnk(unk);

	std::vector<std::shared_ptr<std::thread>> threads(thread);
	std::vector<bool> results(thread);

	std::cout << "cost     progress[token]  speed [token/sec]" << std::endl;

	for (unsigned int e = 0; e < epoch; ++e)
	{
		std::cout << "Epoch " << e + 1 << std::endl;
		total_processed_words = 0;
		total_cost = 0;
		for (size_t t = 0; t < thread; ++t)
		{
			threads[t].reset(new std::thread([&results](size_t id)
			{
				results[id] = ThreadProcedure(id);
			}, t));
		}
		std::thread counter_thread([]()
			{
				typedef std::chrono::steady_clock SClock;

				auto start = SClock::now();
				std::chrono::duration<double> diff;

				size_t former = 0;
				while (!terminated)
				{
					std::this_thread::sleep_for(std::chrono::seconds(1));
					fprintf(stdout, "\r%-8g %-16zu %-16zu",
							total_cost / total_processed_words,
							total_processed_words,
						(size_t)(double(total_processed_words - former) / (diff = SClock::now() - start).count()));
					
					former = total_processed_words;
					start = SClock::now();
					
					fflush(stdout);
				}
				std::cout << std::endl;
			});
		for (auto& t : threads)
			t->join();
		terminated = true;
		counter_thread.join();

		const bool result = std::accumulate(results.begin(), results.end(), true, std::logical_and<bool>());

		if (!result)
			break;
		
		std::cerr << "Writing model \"" << output_filename << "\" ... ";
		std::cerr.flush();
		if (WriteModel(output_filename))
			std::cerr << "Done" << std::endl;
		else
			std::cerr << "Failed!" << std::endl;
	}
	delete[] M;
	delete[] gradsq;
	std::cerr << "Done" << std::endl;
	return 0;
}
