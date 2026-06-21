#include "tokenizer.h"

#include <cctype>

namespace cpp_bm25 {

Tokenizer::Tokenizer(bool remove_stop_words) : remove_stop_words_(remove_stop_words) {
	if (remove_stop_words_) {
		InitDefaultStopWords();
	}
}

std::vector<std::string> Tokenizer::Tokenize(const std::string& text) const {
	std::vector<std::string> tokens;
	std::string current;
	
	for (char ch : text) {
		if (std::isalnum(static_cast<unsigned char>(ch))) {
			current += static_cast<char>(
				std::tolower(static_cast<unsigned char>(ch)));
		} else {
			// Non-alphanumeric character acts as a delimiter.
			if (!current.empty()) {
				if (!remove_stop_words_ || stop_words_.find(current) == stop_words_.end()) {
					tokens.push_back(current);
				}
				current.clear();
			}
		}
	}

	// Handle the last token if the text does not end with a delimiter.
	if (!current.empty()) {
		if (!remove_stop_words_ || stop_words_.find(current) == stop_words_.end()) {
			tokens.push_back(current);
		}
	}

	return tokens;
}

void Tokenizer::AddStopWord(const std::string& word) {
	stop_words_.insert(ToLower(word));
}

void Tokenizer::RemoveStopWord(const std::string& word) {
	stop_words_.erase(ToLower(word));
}

void Tokenizer::InitDefaultStopWords() {
	// Standard English stop word list.
	static const char* words[] = {
		"a",       "about",    "above",   "after",   "again",   "against",
		"all",     "am",       "an",      "and",     "any",     "are",
		"aren't",  "as",       "at",      "be",      "because", "been",
		"before",  "being",    "below",   "between", "both",    "but",
		"by",      "can't",    "cannot",  "could",   "couldn't","did",
		"didn't",  "do",       "does",    "doesn't", "doing",   "don't",
		"down",    "during",   "each",    "few",     "for",     "from",
		"further", "had",      "hadn't",  "has",     "hasn't",  "have",
		"haven't", "having",   "he",      "he'd",    "he'll",   "he's",
		"her",     "here",     "here's",  "hers",    "herself", "him",
		"himself", "his",      "how",     "how's",   "i",       "i'd",
		"i'll",    "i'm",      "i've",    "if",      "in",      "into",
		"is",      "isn't",    "it",      "it's",    "its",     "itself",
		"let's",   "me",       "more",    "most",    "mustn't", "my",
		"myself",  "no",       "nor",     "not",     "of",      "off",
		"on",      "once",     "only",    "or",      "other",   "ought",
		"our",     "ours",     "ourselves","out",    "over",    "own",
		"same",    "shan't",   "she",     "she'd",   "she'll",  "she's",
		"should",  "shouldn't","so",      "some",    "such",    "than",
		"that",    "that's",   "the",     "their",   "theirs",  "them",
		"themselves","then",   "there",   "there's", "these",   "they",
		"they'd",  "they'll",  "they're", "they've", "this",    "those",
		"through", "to",       "too",     "under",   "until",   "up",
		"very",    "was",      "wasn't",  "we",      "we'd",    "we'll",
		"we're",   "we've",    "were",    "weren't", "what",    "what's",
		"when",    "when's",   "where",   "where's", "which",   "while",
		"who",     "who's",    "whom",    "why",     "why's",   "with",
		"won't",   "would",    "wouldn't","you",     "you'd",   "you'll",
		"you're",  "you've",   "your",    "yours",   "yourself","yourselves"
	};

	for (const char* w : words) {
		stop_words_.insert(std::string(w));
	}
}

std::string Tokenizer::ToLower(const std::string& s) {
	std::string result;
	result.reserve(s.size());
	for (char ch : s) {
		result += static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
	}
	return result;
}

} // namespace cpp_bm25