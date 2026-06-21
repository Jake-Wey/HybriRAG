#pragma once

#include "inverted_index.h"
#include "tokenizer.h"

#include <cstdint>
#include <string>
#include <vector>
#include <shared_mutex>
#include <mutex>

namespace cpp_bm25 {

/// <summary>
/// BM25 scoring engine.
/// </summary>
class BM25Engine {
public:
	/// <summary>
	/// Construct a BM25 engine with default parameters (k1=1.5, b=0.75).
	/// </summary>
	BM25Engine();

	/// <summary>
	/// Construct a BM25 engine with custom k1 and b parameters.
	/// </summary>
	/// <param name="k1">k1</param>
	/// <param name="b">b</param>
	BM25Engine(double k1, double b);

	/// <summary>
	/// Add a single document to the engine.
	/// </summary>
	/// <param name="doc_id">doc_id</param>
	/// <param name="text">doc content</param>
	void AddDocument(int64_t doc_id, const std::string& text);

	/// <summary>
	/// Add multiple documents at once.
	/// </summary>
	/// <param name="doc_ids">doc ids</param>
	/// <param name="texts">docs content</param>
	void AddDocuments(const std::vector<int64_t>& doc_ids,
					 const std::vector<std::string>& texts);

	/// <summary>
	/// Query the engine and return the top_k results sorted by descending
	/// </summary>
	/// <param name="query_text">query</param>
	/// <param name="top_k">top k</param>
	/// <returns>a vector of (doc_id, score) pairs.</returns>
	std::vector<std::pair<int64_t, double>> Query(const std::string& query_text,
												  int64_t top_k = 10) const;

	/// <summary>
	/// Return the number of documents currently in the index
	/// </summary>
	/// <returns>number of documents</returns>
	int64_t GetDocumentCount() const;

	/// <summary>
	/// Reset the engine, removing all documents.
	/// </summary>
	void Clear();

	double GetK1() const { return k1_; }
	double GetB() const { return b_; }

private:
	double k1_;
	double b_;

	InvertedIndex index_;
	Tokenizer tokenizer_;

	//  Protects index_ modifications; queries use the index's own shared_mutex.
	mutable std::shared_mutex write_mutex_;

	/// <summary>
	/// Compute the IDF
	/// </summary>
	/// <param name="term">target term</param>
	/// <returns>IDF value</returns>
	double ComputeIDF(const std::string& term) const;

	/// <summary>
	/// Compute the BM25 score for a single document against the given query
	/// </summary>
	/// <param name="doc_id">doc id</param>
	/// <param name="query_tokens"> query tokens</param>
	/// <returns>score</returns>
	double ScoreDocument(int64_t doc_id,
						 const std::vector<std::string>& query_tokens) const;
};

} // namespace cpp_bm25