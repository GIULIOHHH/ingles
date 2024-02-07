Digit sums
```python
def sum_digits3(n):
   r = 0
   while n:
       r, n = r + n % 10, n // 10
   return r
```


SEGMENT TREE

# Solve static range sums
```python
n,q=map(int,input().split())
l=list(map(int,input().split()))

def solve(index,l_cur,r_cur):
    #if out of bounds
    if l_cur>r_query or r_cur<l_query:
        return 0
    #if completly contained
    if l_query<=l_cur and r_cur<=r_query:
        return tree[index]
    #else get the sum of the 2 children
    mid=(l_cur+r_cur)//2
    left_child=solve(index*2,l_cur,mid)
    right_child=solve(index*2+1,mid+1,r_cur)
    return left_child+right_child
    

#pad w 0s to get a power of 2

if n==0:
    powr=1
else:
    powr=2**(n - 1).bit_length()
l+=[0]*(powr-n)  


#build the segment tree
tree=[None]*powr
tree+=l
for i in reversed(range(1,powr)):
    tree[i]=tree[i*2]+tree[i*2+1]

# process queries
for _ in range(q):
    l_query,r_query=map(int,input().split())
    print(solve(1,1,powr))

```

##### in c++
```c++
#include <iostream>
#include <vector>
#include <cmath>
 
using namespace std;
 
class SegmentTree {
private:
    vector<long long> tree;
    int n;
 
    long long solve(int index, int l_cur, int r_cur, int l_query, int r_query) {
        // If out of bounds
        if (l_cur > r_query || r_cur < l_query) {
            return 0;
        }
        // If completely contained
        if (l_query <= l_cur && r_cur <= r_query) {
            return tree[index];
        }
        // Else get the sum of the two children
        int mid = (l_cur + r_cur) / 2;
        long long left_child = solve(index * 2, l_cur, mid, l_query, r_query);
        long long right_child = solve(index * 2 + 1, mid + 1, r_cur, l_query, r_query);
        return left_child + right_child;
    }
 
public:
    SegmentTree(const vector<long long>& l) {
        int powr = pow(2, ceil(log2(l.size())));
        n = powr;
        tree.resize(2 * powr, 0);
        for (int i = 0; i < l.size(); ++i) {
            tree[powr + i] = l[i];
        }
        for (int i = powr - 1; i > 0; --i) {
            tree[i] = tree[i * 2] + tree[i * 2 + 1];
        }
    }
 
    long long query(int l_query, int r_query) {
        return solve(1, 1, n, l_query, r_query);
    }
};
 
int main() {
    int n, q;
    cin >> n >> q;
    vector<long long> l(n);
    for (int i = 0; i < n; ++i) {
        cin >> l[i];
    }
 
    SegmentTree segmentTree(l);
 
    for (int i = 0; i < q; ++i) {
        int l_query, r_query;
        cin >> l_query >> r_query;
        cout << segmentTree.query(l_query, r_query) << endl;
    }
 
    return 0;

```


# Updating the tree

```python
#update the tree
def update(index,newval):
    #change the leaf
    newindex=powr-1+index
    tree[newindex]=newval
    #update the rest of the tree
    newindex=newindex//2
    while newindex>=1:
        tree[newindex]=tree[newindex*2]+tree[newindex*2+1]
        newindex=newindex//2
```

XOR 
```c++
int basis[d]; // basis[i] keeps the mask of the vector whose f value is i

int sz; // Current size of the basis

void insertVector(int mask) {
	for (int i = 0; i < d; i++) {
		if ((mask & 1 << i) == 0) continue; // continue if i != f(mask)

		if (!basis[i]) { // If there is no basis vector with the i'th bit set, then insert this vector into the basis
			basis[i] = mask;
			++sz;
			
			return;
		}

		mask ^= basis[i]; // Otherwise subtract the basis vector from this vector
	}
}
```


HASHING

  
Sure! Let's demonstrate how the hash comparison would work for the strings `a = "01011010"` and `b = "10101110"` using the polynomial rolling hash method. We'll assume the `base` is 3 and `mod` is a large prime number, say 1000000007 (10^9 + 7), as in your original C++ code.

### Step 1: Converting Characters to Numbers

- Since the strings consist of binary digits ('0' and '1'), we can use their ASCII values for conversion, or simply use `0` for '0' and `1` for '1'.

### Step 2: Computing Prefix Hashes

- We will compute the hash for each prefix of the strings. The hash of a prefix of length `n` is calculated as:
    `hash = (char[0] * base^(n-1) + char[1] * base^(n-2) + ... + char[n-1] * base^0) % mod`
- Let's do this for both strings `a` and `b`.

### Step 3: Comparing Substrings

- After computing these hashes, we can compare any substring of a given length by using the hash values. The hash of a substring from index `i` to `j` can be computed using the formula:

	`hash(substring(i, j)) = (hash(prefix(j)) - hash(prefix(i)) * base^(j-i)) % mod`

- For our example, let’s compare substrings of length 3 starting from index 2 in both strings.


2 FINGER SORTING

```python
while q<ln and j<ln:
        a=tree[i*2][q]
        b=tree[i*2+1][j]
        
        if a<b:
            #we can optimize cur
            cur*=a
            pref[i].append(cur)
            tree[i].append(a)
            q+=1
        else:
            cur*=b
            pref[i].append(cur)
            tree[i].append(b)
            j+=1
    while q<ln:
        a=tree[i*2][q]
        cur*=a
        pref[i].append(cur)
        tree[i].append(a)
        q+=1
    while j<ln:
        b=tree[i*2+1][j]
        cur*=b
        pref[i].append(cur)
        tree[i].append(b)
        j+=1

```



MATRIX EXPONENTIATION

```python
MOD = 10**9 + 7  # Define the modulus

def matrix_multiply(A, B, MOD):
    """
    Performs matrix multiplication with modulus.
    Args:
        A, B: Matrices to be multiplied
        MOD: Modulus value for modular arithmetic
    Returns:
        Resultant matrix after multiplication
    """
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    #CREATES AN EMPTY MATRIX
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

	"""
	PERFORMS MATRIX MULTIPLICATION
	"""
    for i in range(rows_A): 
        for j in range(cols_B):
            for k in range(cols_A): 
                result[i][j] = (result[i][j] + A[i][k] * B[k][j]) % MOD
    return result

def matrix_power(M, n, MOD):
    """
    Computes the matrix M to the power of n using binary exponentiation with modulus.
    Args:
        M: Matrix to be exponentiated
        n: Exponent
        MOD: Modulus value for modular arithmetic
    Returns:
        Matrix raised to the power n
    """
    #creates an identity matrix of len(M)
    result = [[1 if i == j else 0 for j in range(len(M))] for i in range(len(M))]
    while n > 0:
        if n % 2:  # If n is odd, multiply result by M
            result = matrix_multiply(result, M, MOD)
        M = matrix_multiply(M, M, MOD)  # Square the matrix M
        n //= 2  # Divide n by 2
    return result

# Define the transformation matrix
transformation_matrix = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]

# Example exponent
n = 67

# Calculate the matrix power with modulus
M_n_mod = matrix_power(transformation_matrix, n, MOD)


```
