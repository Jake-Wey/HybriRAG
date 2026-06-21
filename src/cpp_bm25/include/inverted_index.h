#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <shared_mutex>
#include <mutex>

namespace cpp_bm25 {

/// <summary>
/// A single posting entry: records a document ID, the term frequency within
/// that document, and the positional offsets where the term occurs.
/// </summary>
struct Posting {
	int64_t doc_id;
	int64_t term_freq;
	std::vector<int64_t> positions;
};

/// <summary>
/// Inverted index that maps terms to their posting lists.
/// </summary>
class InvertedIndex {
public:
	InvertedIndex() = default;

	/// <summary>
	/// Add a posting for the given term.
	/// </summary>
	/// <param name="term">target term</param>
	/// <param name="posting">target posting</param>
	void AddPosting(const std::string& term, const Posting& posting);

	/// <summary>
	/// Retrieve the posting list for a term.
	/// </summary>
	/// <param name="term">target term</param>
	/// <returns>list of posting</returns>
	std::vector<Posting> GetPostings(const std::string& term) const;

	/// <summary>
	/// Return the number of distinct documents that contain the given term.
	/// </summary>
	/// <param name="term">target term</param>
	/// <returns>number of the docs that contain the term</returns>
	int64_t GetDocumentFrequency(const std::string& term) const;

	/// <summary>
	/// Return the total number of documents in the index.
	/// </summary>
	/// <returns>doc number</returns>
	int64_t GetDocumentCount() const { return doc_count_; }

	/// <summary>
	/// Set the total document count (used after bulk inserts).
	/// </summary>
	/// <param name="count">doc count</param>
	void SetDocumentCount(int64_t count) { doc_count_ = count; }

	/// <summary>
	/// Increment the total document count by 1.
	/// </summary>
	void IncrementDocumentCount() { ++doc_count_; }

	/// <summary>
	/// Return the average document length across all documents.
	/// </summary>
	/// <returns>average length</returns>
	double GetAverageDocumentLength() const;

	/// <summary>
	/// Set the total length for average computation.
	/// </summary>
	/// <param name="length">length</param>
	void SetTotalDocumentLength(int64_t length) {
		total_doc_length_ = length;
	}

	/// <summary>
	/// Add `length` to the running total of document lengths.
	/// </summary>
	/// <param name="length">length</param>
	void AddTotalDocumentLength(int64_t length) {
		total_doc_length_ += length;
	}

	/// <summary>
	/// Return the length of a specific document (0 if unknown).
	/// </summary>
	/// <param name="doc_id">document id</param>
	/// <returns>doc length</returns>
	int64_t GetDocumentLength(int64_t doc_id) const;

	/// <summary>
	/// Set the length of a specific document.
	/// </summary>
	/// <param name="doc_id">document id</param>
	/// <param name="length">length</param>
	void SetDocumentLength(int64_t doc_id, int64_t length);

	/// <summary>
	/// Clear the entire index.
	/// </summary>
	void Clear();

	/// <summary>
	/// Return true if the index contains no postings.
	/// </summary>
	/// <returns>if the index contains no postings</returns>
	bool Empty() const;

private:
	int64_t doc_count_ = 0;
	int64_t total_doc_length_ = 0;

	// term -> list of postings
	std::unordered_map<std::string, std::vector<Posting>> index_;

	// doc_id -> document length (in tokens)
	std::unordered_map<int64_t, int64_t> doc_lengths_;

	// Protects index_, doc_lengths_, doc_count_, total_doc_length_
	mutable std::shared_mutex mutex_;
};

} // namespace cpp_bm25