#include <bits/stdc++.h>
using namespace std;


class Solution {
    public:
        int x, y;
        int dpcall(vector<vector<int>>& matrix, int i, int j, vector<vector<int>> &dp){
            //i is for x and j is for y
            if(i < 0 || j < 0 || !check(i, j)){return 0;}
            if(dp[j][i] != -1){
                int ans = 0;
                //we start checking for all the dimensions;
                int l, r, u, d;
                l = dpcall(matrix, i - 1, j, dp);
                r = dpcall(matrix, i + 1, j, dp);
                u = dpcall(matrix, i, j - 1, dp);
                d = dpcall(matrix, i, j + 1, dp);
                
                //we have to check if the adjacent elemtns are smaller than the current one;
                if(check(i, j - 1) && matrix[j][i] > matrix[j - 1][i]){ans = max(ans, l);}
                if(check(i, j + 1) && matrix[j][i] > matrix[j + 1][i]){ans = max(ans, r);}
                if(check(i - 1, j) && matrix[j][i] > matrix[j][i - 1]){ans = max(ans, u);}
                if(check(i + 1, j) && matrix[j][i] > matrix[j][i + 1]){ans = max(ans, d);}
                dp[j][i] = ans + matrix[j][i];
            }
            return dp[j][i];
        }
        int check(int a, int b){
            if(a < x && b < y){return 1;}else return 0; }
        int longestIncreasingPath(vector<vector<int>>& matrix) {
            y = matrix.size();
            x = matrix[0].size();

            vector<vector<int>>dp(y, vector<int>(x, -1));
            return dpcall(matrix, x - 1, y - 1, dp);
        }
};