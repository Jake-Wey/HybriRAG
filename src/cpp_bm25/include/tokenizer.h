#pragma once

#include <string>
#include <vector>
#include <unordered_set>

namespace cpp_bm25 {

/// <summary>
/// Simple tokenizer that lowercases text, splits on whitespace and 
/// punctuation, and optionally removes stop words.
/// </summary>
class Tokenizer {
public:
	/// <summary>
	/// Construct a tokenizer.
	/// </summary>
	/// <param name="remove_stop_words">If true, common English stop words are
	/// filtered out.
	/// </param>
	Tokenizer(bool remove_stop_words = true);

	/// <summary>
	/// Splits on any non-alphanumeric character, removes empty tokens, and
	/// optionally filters stop words.
	/// </summary>
	/// <param name="text">given text</param>
	/// <returns>a list of lowercase tokens</returns>
	std::vector<std::string> Tokenize(const std::string& text) const;

	/// <summary>
	/// Add a custom stop word.
	/// </summary>
	/// <param name="word">a custom stop word.</param>
	void AddStopWord(const std::string& word);

	/// <summary>
	/// Remove a stop word
	/// </summary>
	/// <param name="word">a stop word</param>
	void RemoveStopWord(const std::string& word);

	/// <summary>
	/// Return the current set of stop words.
	/// </summary>
	/// <returns>set of stop words.</returns>
	const std::unordered_set<std::string>& GetStopWords() const {
		return stop_words_;
	}
private:
	std::unordered_set<std::string> stop_words_;
	bool remove_stop_words_;

	/// <summary>
	/// Initialize the default English stop-word list.
	/// </summary>
	void InitDefaultStopWords();

	/// <summary>
	/// Convert a string to lowercase (ASCII only).
	/// </summary>
	/// <param name="s">input string</param>
	/// <returns>lowercase string</returns>
	static std::string ToLower(const std::string& s);
};

} // namespace cpp_bm25