def i_may_be_wrong(model, X, y_expected, min, max):
    y_pred = model.predict(X).round()

    correct_results = 0
    answers = 0
    n = len(y_pred)
    for i in range(n):
        p = y_pred[i][0]
        if (p <= min) or (p >= max):
            answers += 1
            if p == y_expected[i][0]:
                correct_results += 1

    return answers/n, correct_results/answers


def i_may_be_wrong_categorical(model, X, y_expected, threshold):
    y_pred = model.predict(X)

    correct_results = 0
    answers = 0
    n = len(y_pred)
    for i in range(n):
        p = y_pred[i]
        if p.max() >= threshold:
            answers += 1
            if (p.round() == y_expected[i]).all():
                correct_results += 1

    return answers / n, correct_results / answers
