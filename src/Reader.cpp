#include <algorithm>
#include <utility>
#include <sstream>

#include "Reader.h"

Reader::Reader(FILE* fin)
	: state(OUTW), fin(fin), pos(0)
{
	if (fin == NULL)
		state = EOF_S;
}

bool Reader::ReadNext()
{
	int ch;
	word.clear();

	pos += 1;

	for (ch = fgetc(fin); ; ch = fgetc(fin), ++pos)
	switch (state)
	{
		case INW:
			switch (ch)
			{
			case ' ': case '\t': case '\r':
				state = OUTW;
				return true;
			case '\v': case '\n': case '\f':
			case EOF:
				state = EOS_P;
				ungetc(ch, fin);
				--pos;
				return true;
			default:
				word.insert(word.end(), ch);
				break;
			}
			break;
		case OUTW:
			switch (ch)
			{
			case ' ': case '\t': case '\r':
				break;
			case '\v': case '\n': case '\f':
				state = EOS;
				return true;
			case EOF:
				state = EOF_S;
				return false;
			default:
				word.insert(word.end(), ch);
				state = INW;
				break;
			}
			break;
		case EOS:
			switch (ch)
			{
			case ' ': case '\t': case '\r':
			case '\v': case '\n': case '\f':
				break;
			case EOF:
				state = EOF_S;
				return false;
			default:
				word.insert(word.end(), ch);
				state = INW;
				break;
			}
			break;
		case EOS_P:
			state = EOS;
			return true;
		case EOF_S:
			return false;
	}
}

const std::string& Reader::GetToken() const
{
	return word;
}

const size_t& Reader::GetPosition() const
{
	return pos;
}

void Vocabulary::Read(std::istream& fin)
{
	std::string line, word;
	while (std::getline(fin, line))
	{
		std::istringstream iss(line);
		iss >> word;
		insert(word);
	}
}

Vocabulary::Vocabulary()
{
}

size_t Vocabulary::GetIndexInvalid(const std::string& w) const
{
	const auto where = _w2i.find(w);
	return where == _w2i.end() ? _w2i.size() : where->second;
}

size_t Vocabulary::GetIndexFallback(const std::string & word) const
{
	const auto where = _w2i.find(word);
	return where == _w2i.end() ? _unk : where->second;
}

void Vocabulary::clear()
{
	_w2i.clear();
	_i2w.clear();
}

size_t Vocabulary::size() const
{
	return _w2i.size();
}

void Vocabulary::SetUnk(const std::string & unk)
{
	if (_w2i.find(unk) != _w2i.end())
		_unk = _w2i.find(unk)->second;
	else
		_unk = _w2i.size();
}

void Vocabulary::insert(const std::string & word)
{
	const auto n = _w2i.size();
	_i2w[n] = word;
	_w2i[word] = n;
}

size_t Vocabulary::GetIndex(const std::string& w)
{
	const auto where = _w2i.find(w);
	if (where == _w2i.end())
	{
		const auto n = _w2i.size();
		insert(w);
		return n;
	}else
		return where->second;
}

const std::string& Vocabulary::GetWord(size_t i, const std::string & unk) const
{
	auto const where = _i2w.find(i);
	if (where == _i2w.end())
		return unk;
	else
		return where->second;
}

WindowsReader::WindowsReader(Reader & reader, const Vocabulary & vocab, unsigned int window_l, unsigned int window_r)
	: _reader(reader), _vocab(vocab), _word(), _window_l(window_l), _window_r(window_r)
{
	BeginSentence();
}

bool WindowsReader::ReadItem()
{
	if (IsSentenceBoundary())
	{
		BeginSentence();
		return true;
	}
	if (!_reader.GetToken().empty()) // not EOS
		GetNext();

	if (_word == _window_l)
	{
		_discourse.erase(_discourse.begin());
		_positions.erase(_positions.begin());
	}
	else
		++_word;
	return false;
}

bool WindowsReader::IsGood() const
{
	return !_discourse.empty();
}

bool WindowsReader::IsSentenceBoundary() const
{
	return _word + 1 == _discourse.size();
}

bool WindowsReader::GetNext()
{
	auto const result = _reader.ReadNext();
	if (result)
	{
		if (!_reader.GetToken().empty())
		{
			_discourse.emplace_back(_vocab.GetIndexFallback(_reader.GetToken()));
			_positions.emplace_back(_reader.GetPosition());
		}
	}
	return result;
}

void WindowsReader::BeginSentence()
{
	_discourse.clear();
	_positions.clear();
	_word = 0;
	while (_discourse.size() <= _window_r && GetNext())
	{
		_word = std::max<long long>(0, _discourse.size() - 1 - _window_r);
		if (_reader.GetToken().empty()) // EOS
			return;
	}
}
