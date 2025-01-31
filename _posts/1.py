def solution(array, commands):
    answer = []
    for command in commands:
        sliced_array = array[command[0]-1:command[1]]
        print("sliced_array ", sliced_array)
        sliced_array.sort()
        print("sorted_array ", sliced_array)
        answer.append(sliced_array[command[2]-1])
        print("answer ", answer)
    return answer

solution([1, 5, 2, 6, 3, 7, 4],	[[2, 5, 3], [4, 4, 1], [1, 7, 3]])