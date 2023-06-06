import os
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

student_files = [doc for doc in os.listdir() if doc.endswith('.txt')]
student_notes = [open(_file, encoding='utf-8').read() for _file in student_files]

vectorizer = HashingVectorizer(binary=True)
vectors = vectorizer.fit_transform(student_notes).toarray()
s_vectors = list(zip(student_files, vectors))
plagiarism_results = set()

def check():
    global s_vectors
    if len(s_vectors) < 2:
        return
    for student_1,text_vector_1 in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((student_1,text_vector_1))
        del new_vectors[current_index]
        for student_2, text_vector_2 in new_vectors:
            sim_score=cosine_similarity([text_vector_1,text_vector_2])[0][1]
            student_pair = sorted((student_1,student_2))
            score=(student_pair[0],student_pair[1], sim_score)
            plagiarism_results.add(score)
    return plagiarism_results

plagiarism_results=check()

if plagiarism_results:

    for data in plagiarism_results:
        print(data)
else:
    print("No plagiarism")
