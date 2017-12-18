#ifndef INCLUDE_SPARSITY_H
#define  INCLUDE_SPARSITY_H

#include <utility>
#include <unordered_map>
#include <vector>

class Sparsity
{
public:
	typedef std::pair<size_t, size_t> Position;
	typedef std::vector<Position> Positions;
private:
	class hash
	{
	public:
		size_t operator() (const Position& p)const;
	};
	//! index triplet
	/*!
	to obtain the matrix dot product: v3 = v1.v2
	v1[i] * v2[j] -> v3[k]
	*/
	struct Indices
	{
		Indices(size_t i, size_t j, size_t k)
			: i(i), j(j), k(k)
		{}
		size_t i, j, k;
	};

	typedef std::vector<Indices> Kernel;
public:
	void InitKernels(unsigned int dim, unsigned int stripes, bool sym);
	void InitKernels(const std::vector<std::vector<bool>>& mask);
	void InitKernels(const Positions& positions);
	Sparsity();
	~Sparsity();
	
	size_t GetParameterNumber()const;
	size_t GetDim()const;
	
	template<class Precision>
	void DotProduct(const Precision* left_m, const Precision* right_m, Precision* result_m)const
	{
		return Product(left_m, right_m, result_m, _dotKernel);
	}
	template<class Precision>
	void BackPropagate(const Precision* left_m, const Precision* right_m, Precision* result_m)const
	{
		return Product(left_m, right_m, result_m, _backPropKernel);
	}
	template<class Precision>
	Precision Sandwitch(const Precision* left_m, const Precision* middle_m, const Precision* right_v)const
	{
		Precision result = 0;
		for (const auto& index : _sandwitchKernel)
			result += left_m[index.i] * middle_m[index.j] * right_v[index.k];
		return result;
	}

	const std::vector<size_t>& GetOnes();

	bool IsAssociative();

	typedef std::unordered_map<Position, size_t, hash> Pos2iType;

	const Pos2iType& GetPositions()const { return _pos2i; }

	static Positions ParseLine(const std::string& line);

private:
	template<class Precision>
	void Product(const Precision* left, const Precision* right, Precision* result, const Kernel& kernel)const
	{
		for (const auto& index : kernel)
			result[index.k] += left[index.i] * right[index.j];
	}

	Kernel _sandwitchKernel; //!< ones * matrix * matrix * matrix * ones -> scalar ( param * param * param -> 1)
	Kernel _dotKernel;       //!< matrix * matrix -> matrix ( param * param -> param)
	Kernel _backPropKernel;  //!< (M1, M3) -> Derivative(ones * M1 * M2 * M3 * ones, M2)

	std::vector<size_t> _ones;  //!< contains the positions where the identity matrix should contain 1s (0 elsewhere)

	Pos2iType _pos2i;

	void InitMask(unsigned int dim, unsigned int stripes, bool sym);
	void InitMask(const std::vector<std::vector<bool>>& mask);
	void InitKernels();

	bool IsInMask(const Position& x);
	bool IsInMask(const Position::first_type& i, const Position::second_type& j);

	size_t _dim;
};

#endif // !INCLUDE_SPARSITY_H
