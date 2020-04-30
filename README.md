# USAGE
## Arguments
1. input directory
    * contains the images(.jpg) to be detected
2. output directory
    * the output result directory
    * need to be created in advance
3. binary_threshold
    * threshold for global binarization
4. sba_size
    * the size of small bright area needs to be removed
5. r1
    * parameter of morphological closing
6. r2
    * parameter of morphological closing
7. center_width
    * the width of intermediate calculation area
    * default = 1500 um
8. niblack_n
    * parameter of niblack algorithm
9. niblack_k
    * parameter of niblack algorithm

## Example

```shell
python ChoroidalBoundaryDetection.py /Users/rick/PycharmProjects/ChoroidalBoundary/data/OCT/ /Users/rick/PycharmProjects/ChoroidalBoundary/data/OCT_res/ 20 10 20 25 1500 13 0.1
```