#include "Sparsity.h"

#include <set>
#include <array>
#include <sstream>

size_t Sparsity::GetParameterNumber()const
{
	return _pos2i.size();
}

size_t Sparsity::GetDim()const
{
	return _dim;
}

void Sparsity::InitKernels()
{
	_sandwitchKernel.clear();
	_dotKernel.clear();
	_ones.clear();

	_dim = 0;
	for (const auto& pos : _pos2i)
	{
		auto const i = pos.first.first;
		auto const j = pos.first.second;
		auto const k = pos.second;

		_dim = std::max(std::max(_dim, i), j);

		if (i == j)
			_ones.emplace_back(k);
	}

	//dot
	for (const auto& pos1 : _pos2i)
		for (const auto& pos2 : _pos2i)
		{
			auto const i = pos1.first.first;
			auto const k1 = pos1.first.second;
			auto const k2 = pos2.first.first;
			auto const j = pos2.first.second;

			if (k1 == k2 && IsInMask(i, j))
				_dotKernel.emplace_back(pos1.second, pos2.second, _pos2i[Position(i, j)]);
		}
	// backprop
	for (const auto& pos1 : _pos2i)
		for (const auto& pos2 : _pos2i)
			for (const auto& pos3 : _pos2i)
			{
				if (pos1.first.second == pos2.first.first && pos2.first.second == pos3.first.first)
				{
					const auto i = pos1.first.first;
					// const auto j = pos1.first.second;
					const auto k = pos2.first.second;
					const auto l = pos3.first.second;
					if (IsInMask(i, l) && IsInMask(i, k))
					{
						_backPropKernel.emplace_back(pos1.second, pos3.second, pos2.second);
						_sandwitchKernel.emplace_back(pos1.second, pos2.second, pos3.second);
					}
				}
			}

	++_dim;
	//if (!IsAssociative())
	//	fprintf(stderr, " Associator is non-zero! ");
}

void Sparsity::InitKernels(const std::vector<std::vector<bool>>& mask)
{
	InitMask(mask);
	InitKernels();
}

void Sparsity::InitKernels(const Positions & positions)
{
	for (size_t i = 0; i < positions.size(); ++i)
		_pos2i[positions[i]] = i;
	InitKernels();
}

Sparsity::Sparsity()
{
}

Sparsity::~Sparsity()
{
}

void Sparsity::InitKernels(unsigned int dim, unsigned int stripes, bool sym)
{
	InitMask(dim, stripes, sym);
	InitKernels();
}

const std::vector<size_t>& Sparsity::GetOnes()
{
	return _ones;
}

//const std::vector<std::vector<int>>& Sparsity::GetConstraints()
//{
//	return _constraints;
//}

bool Sparsity::IsAssociative()
{
	std::set<std::array<Position, 3>> a1, a2;

	for (const auto& pos1 : _pos2i)
		for (const auto& pos2 : _pos2i)
			for (const auto& pos3 : _pos2i)
			{
				if (pos1.first.second == pos2.first.first && pos2.first.second == pos3.first.first)
				{
					const auto i = pos1.first.first;
					const auto j = pos1.first.second;
					const auto k = pos2.first.second;
					const auto l = pos3.first.second;
					if (IsInMask(i, l))
					{
						std::array<Position, 3> new_pos;
						new_pos[0] = pos1.first; new_pos[1] = pos2.first; new_pos[2] = pos3.first;
						if (IsInMask(i, k))
							a1.insert(new_pos);
						if (IsInMask(j, l))
							a2.insert(new_pos);
					}
				}
			}
	return a1 == a2;
}

Sparsity::Positions Sparsity::ParseLine(const std::string & line)
{
	std::string str = line;
	Sparsity::Position pos;
	Sparsity::Positions positions;
	for (auto it = str.begin(); it != str.end(); ++it)
		if (!('0' <= *it && *it <= '9'))
			*it = ' ';
	std::istringstream iss(str);

	while (iss >> pos.first >> pos.second)
	{
		positions.emplace_back(pos);
	}
	return positions;
}

void Sparsity::InitMask(unsigned int dim, unsigned int stripes, bool sym)
{
	_pos2i.clear();
	_pos2i.rehash(dim*dim);

	size_t parameters = 0;
	for (int i = 0; i < (int)dim; ++i)
		for (int j = 0; j < (int)dim; ++j)
			if ((sym && std::abs(j - i) < (int)stripes) || (!sym && i >= j && i < j + (int)stripes))
			{
				_pos2i[Position(i, j)] = parameters++;
			}
}

void Sparsity::InitMask(const std::vector<std::vector<bool>>& mask)
{
	size_t parameters = 0;
	_pos2i.clear();
	
	for (size_t row = 0; row < mask.size(); ++row)
	{		
		for (size_t col = 0; col < mask[row].size(); ++col)
		{
			if (mask[row][col])
			{
				_pos2i[Position(row, col)] = parameters++;
			}
		}
	}
}

bool Sparsity::IsInMask(const Position& x)
{
	return _pos2i.find(x) != _pos2i.end();
}

bool Sparsity::IsInMask(const Position::first_type& i, const Position::second_type& j)
{
	return IsInMask(Position(i, j));
}

size_t Sparsity::hash::operator()(const Position& p) const
{
	return p.first * 256 + p.second;
}
