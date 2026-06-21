#include "bm25_engine.h"

#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <algorithm>

namespace cpp_bm25 {

BM25Engine::BM25Engine() : k1_(1.5), b_(0.75) {}

BM25Engine::BM25Engine(double k1, double b) : k1_(k1), b_(b){}

void BM25Engine::AddDocument(int64_t doc_id, const std::string& text) {
	
	std::unique_lock<std::shared_mutex> lock(write_mutex_);

	// Tokenize the document.
	std::vector<std::string> tokens = tokenizer_.Tokenize(text);
	int64_t doc_length = static_cast<int64_t>(tokens.size());

	// Build term -> (frequency, positions) map for this document.
	std::unordered_map<std::string, std::pair<int64_t, std::vector<int64_t>>>
		term_info;
	for (int64_t pos = 0; pos < static_cast<int64_t>(tokens.size()); ++pos) {
		auto& info = term_info[tokens[pos]];
		info.first += 1;
		info.second.push_back(pos);
	}

	// Add postings to the inverted index.
	for (const auto& [term, info] : term_info) {
		Posting posting;
		posting.doc_id = doc_id;
		posting.term_freq = info.first;
		posting.positions = info.second;
		index_.AddPosting(term, posting);
	}

	// Update document metadata.
	index_.SetDocumentLength(doc_id, doc_length);
	index_.AddTotalDocumentLength(doc_length);
	index_.IncrementDocumentCount();
}

void BM25Engine::AddDocuments(const std::vector<int64_t>& doc_ids, 
						     const std::vector<std::string>& texts) {

	if (doc_ids.size() != texts.size()) {
		return;
	}

	std::unique_lock<std::shared_mutex> lock(write_mutex_);

	for (size_t i = 0; i < doc_ids.size(); ++i) {
		int64_t doc_id = doc_ids[i];
		const std::string& text = texts[i];

		std::vector<std::string> tokens = tokenizer_.Tokenize(text);
		int64_t doc_length = static_cast<int64_t>(tokens.size());

		std::unordered_map<
			std::string, std::pair<int64_t, std::vector<int64_t>>>
			term_info;
		for (int64_t pos = 0; pos < static_cast<int64_t>(tokens.size());
			++pos) {
			auto& info = term_info[tokens[pos]];
			info.first += 1;
			info.second.push_back(pos);
		}

		for (const auto& [term, info] : term_info) {
			Posting posting;
			posting.doc_id = doc_id;
			posting.term_freq = info.first;
			posting.positions = info.second;
			index_.AddPosting(term, posting);
		}

		index_.SetDocumentLength(doc_id, doc_length);
		index_.AddTotalDocumentLength(doc_length);
		index_.IncrementDocumentCount();
	}
}

std::vector<std::pair<int64_t, double>> BM25Engine::Query(
	const std::string& query_text, int64_t top_k) const {

	// Tokenize the query
	std::vector<std::string> query_tokens = tokenizer_.Tokenize(query_text);

	if (query_tokens.empty() || index_.GetDocumentCount() == 0) {
		return {};
	}

	// Collect the set of candidate documents
	std::unordered_set<int64_t> candidate_docs;
	std::unordered_set<std::string> seen_terms;
	for (const auto& term : query_tokens) {
		if (seen_terms.count(term)) {
			continue;
		}
		seen_terms.insert(term);

		auto postings = index_.GetPostings(term);
		for (const auto& posting : postings) {
			candidate_docs.insert(posting.doc_id);
		}
	}

	// Score each candidate document.
	std::vector<std::pair<int64_t, double>> scored_docs;
	scored_docs.reserve(candidate_docs.size());
	for (int64_t doc_id : candidate_docs) {
		double score = ScoreDocument(doc_id, query_tokens);
		if (score > 0.0) {
			scored_docs.emplace_back(doc_id, score);
		}
	}

	// Sort by descending score.
	std::partial_sort(
		scored_docs.begin(),
		scored_docs.begin() +
		std::min(static_cast<int64_t>(scored_docs.size()), top_k),
		scored_docs.end(),
		[](const std::pair<int64_t, double>& a,
		   const std::pair<int64_t, double>& b) {
				return a.second > b.second;
		});

	// Return only the top_k results.
	int64_t result_count =
		std::min(static_cast<int64_t>(scored_docs.size()), top_k);
	scored_docs.resize(static_cast<size_t>(result_count));

	return scored_docs;
}

int64_t BM25Engine::GetDocumentCount() const {
	return index_.GetDocumentCount();
}

void BM25Engine::Clear() {
	std::unique_lock<std::shared_mutex> lock(write_mutex_);
	index_.Clear();
}

double BM25Engine::ComputeIDF(const std::string& term) const {
	
	int64_t N = index_.GetDocumentCount();
	int64_t n = index_.GetDocumentFrequency(term);

	// IDF = log((N - n + 0.5) / (n + 0.5) + 1)
	double numerator = static_cast<double>(N) - static_cast<double>(n) + 0.5;
	double denominator = static_cast<double>(n) + 0.5;
	return std::log(numerator / denominator + 1.0);
}

double BM25Engine::ScoreDocument(
	int64_t doc_id, 
	const std::vector<std::string>& query_tokens) const {
	
	double score = 0.0;
	double avgdl = index_.GetAverageDocumentLength();
	double dl = static_cast<double>(index_.GetDocumentLength(doc_id));

	// Track which terms are already scored to avoid double-counting
	// duplicate query terms.
	std::unordered_set<std::string> seen_terms;

	for (const auto& term : query_tokens) {
		if (seen_terms.count(term)) {
			continue;
		}
		seen_terms.insert(term);

		// Look up the term frequency for this document.
		auto postings = index_.GetPostings(term);
		int64_t tf = 0;
		for (const auto& posting : postings) {
			if (posting.doc_id == doc_id) {
				tf = posting.term_freq;
				break;
			}
		}

		if (tf == 0) {
			continue; // Term not in this document.
		}

		double idf = ComputeIDF(term);

		//   IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
		double tf_component =
			(static_cast<double>(tf) * (k1_ + 1.0)) /
			(static_cast<double>(tf) +
			 k1_ * (1.0 - b_ + b_ * dl / avgdl));

		score += idf * tf_component;
	}

	return score;
}

} // namespace cpp_bm25