#ifndef INCLUDE_READER_H
#define INCLUDE_READER_H

#include <stdio.h>

#include <unordered_map>
#include <tuple>
#include <vector>
#include <string>
#include <list>
#include <iostream>

//! Reads a file token-by-token
class Reader
{
public:
	Reader(FILE*);
	//! steps the iteration
	/*!
		@return false if EOF is reached, otherwise true
	*/
	bool ReadNext();
	//! empty word means the end-of-sentence symbol
	const std::string& GetToken()const;
	//! How many bytes were consumed so far
	const size_t& GetPosition()const;
private:
	enum State
	{
		INW,  //!< inside a word
		OUTW, //!< outside a word
		EOS,  //!< end-of-sentence
		EOF_S,//!< end-of-file symbol has been reached
		EOS_P //!< end-of-sentence is pending, next call will result EOS symbol
	} state;
	FILE* const fin;
	std::string word;
	size_t pos;
};

class Vocabulary
{
public:
	Vocabulary();
	//! reads a vocabulary file into a Vocabulary instance
	void Read(std::istream& fin);
	
	const std::string& GetWord(size_t i, const std::string& unk = "<UNK>")const;

	//! get the index of the word from the vocabulary mutable version
	/*!
	@return index of the word. If it is an unseen word
	then it is automatically appended to the vocabulary
	never yields unknown word
	*/
	size_t GetIndex(const std::string& word);

	//! get the index of the word from the vocabulary const version
	/*!
	@return the index if the word or invalid index if the word is not known
	*/
	size_t GetIndexInvalid(const std::string& word)const;

	//! get the index of the word from the vocabulary const version
	/*!
	@return the index if the word or invalid index if the word is not known
	*/
	size_t GetIndexFallback(const std::string& word)const;

	void clear();
	size_t size()const;

	void SetUnk(const std::string& unk);
private:
	void insert(const std::string& word);
	std::unordered_map<std::string, size_t> _w2i;
	std::unordered_map<size_t, std::string> _i2w;
	size_t _unk;

};

class WindowsReader
{
	typedef std::vector<size_t> Type;
public:
	WindowsReader(Reader& reader, const Vocabulary& vocab, unsigned int window_l, unsigned int window_r);
	//! slides the windows with one
	/*!
	@return true if a new sentence have just arrived. Otherwise false.
	*/
	bool ReadItem();
	
	bool IsGood()const;
	const Type& GetContext()const { return _discourse; }
	const size_t& GetWord()const { return _word; }
	const size_t& GetPosition()const { return _positions[_word]; }
private:
	bool IsSentenceBoundary()const;
	bool GetNext();
	Reader& _reader;
	const Vocabulary& _vocab;
	//! contains the whole window
	Type _discourse;
	//! seek position of the file at each word of the discourse
	std::vector<size_t> _positions;
	//! position of the current word in the discourse
	size_t _word;
	const unsigned int _window_l;
	const unsigned int _window_r;

	void BeginSentence();
};

#endif // !INCLUDE_READER_H
