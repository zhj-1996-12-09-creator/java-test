package com.company;

import java.lang.reflect.Method;
import java.util.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.stream.Collectors;

public class TwoNum {

    /**
     * 给定一个整形数组和一个目标和从数组中找两个数字相加等于目标和，输出这两个数字的下标。
     * @param nums
     * @param b
     * @return
     */
    public int[] twoSum(int [] nums , int b){
        Map<Integer,Integer> map = new HashMap<>();
        for (int i = 0 ;i<nums.length;i++){
            if (map.containsKey(b-nums[i])){
                return new int[]{
                        map.get(b-nums[i]),i
                };
            }
            map.put(nums[i],i);
        }
        return null;
    }

    public int[] twoSum2(int [] nums , int b){

        //双层循环
        for (int i = 0; i < nums.length-1 ; i++) {
            for (int j = i+1; j < nums.length; j++) {
                if (nums[j]+nums[i] == b){
                    return new int[]{i,j};
                }
            }
        }
        return null;

    }

    /**两数相加
     * 给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，
     * 并且每个节点只能存储 一位 数字。
     * 请你将两个数相加，并以相同形式返回一个表示和的链表。
     * 你可以假设除了数字 0 之外，这两个数都不会以 0 开头。
     *
     * @param root
     * @return
     */
    ListNode root = null;
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        //使用两个链表表示两个非负整数，再相加
        //那就像加法运算那样，从个位开始，按位相加，注意是按照逆序方式存储的，所以说234，表示4--3---2
        //迭代
        root = new ListNode(l1.val+l2.val);
        ListNode temp = root;
        boolean b = false;
        while (l1 != null || l2 != null ){
            int a = 0;
            int x = (l1 == null) ? 0 : l1.val;
            int y = (l2 == null) ? 0 : l2.val;
            if(b){
                a= x+y+1;
            }else{
                a= x+y;
            }

            if (a>9){
                temp.next = new ListNode(a-10);
                b = true;
            }else{
                temp.next = new ListNode(a);
                b = false;
            }

            if(l1 != null) l1 = l1.next;
            if(l2 != null) l2 = l2.next;
            temp = temp.next;
        }

        //再最后一位如果相加也大于9的话还是要进一
        if(b){
            temp.next = new ListNode(1);
        }
        return root.next;
    }

    /**
     *
     */
    public ListNode addTwoNumbers1(ListNode l1, ListNode l2) {
        //
        if (l1 == null) {
            return l2;
        }
        if (l2 == null) {
            return l1;
        }
        l2.val = l1.val + l2.val;
        if (l2.val >= 10) {
            l2.val = l2.val % 10;
            if (l2.next != null) {
                l2.next.val = l2.next.val + 1;
                if (l2.next.val == 10) {
                    l2.next = addTwoNumbers(new ListNode(0), l2.next);
                }
            } else {
                l2.next = new ListNode(1);
            }
        }
        l2.next = addTwoNumbers(l1.next, l2.next);
        return l2;

    }


    /**
     * 拓展操作：如果给的不是逆序操作，是正序的，则首先需要将链表进行反转，然后再计算
     * 需要两个指针来迭代
     * pre 和 next
     *
     * @param root
     * @return
     */
    public ListNode resverList(ListNode head){
        ListNode pre = null;
        ListNode next = null;
        if (head == null) return null;
        while (head != null){
            next = head.next;
            head.next = pre;
            pre = head;
            head = next;
        }
        return pre;
    }

    /**
     * 递归
     * 首先假设我们实现了将单链表逆序的函数，ListNode reverseListRecursion(ListNode head) ，
     * 、传入链表头，返回逆序后的链表头。
     * 接着我们确定如何把问题一步一步的化小，我们可以这样想。
     * 把 head 结点拿出来，剩下的部分我们调用函数 reverseListRecursion ，这样剩下的部分就逆序了
     * ，接着我们把 head 结点放到新链表的尾部就可以了。这就是整个递归的思想了
     * @param root
     * @return
     */
    public ListNode reverseListRecursion(ListNode head){
        ListNode newHead;
        if(head==null||head.next==null ){
            return head;
        }
        newHead=reverseListRecursion(head.next); //head.next 作为剩余部分的头指针
        head.next.next=head; //head.next 代表新链表的尾，将它的 next 置为 head，就是将 head 加到最后了。
        head.next=null;
        return newHead;
    }


    public boolean isBalanced(TreeNode root) {
        //如果节点是null，默认返回时true
        if (root == null ) return true;
        //如果右子节点不是null，判断是否平衡
        if (root.right != null) isBalanced(root.right);
        //如果左子节点是null，判断是否平衡
        if (root.left != null) isBalanced(root.left);
        int right = getDepth(root.right);
        int left = getDepth(root.left);
        return  Math.abs(right-left)>1 ? false : true;
    }
    static class TreeNode{
        int val;
        int key;
        TreeNode left;
        TreeNode right;
        TreeNode(int x) {
            this.val = x;

        }
    }

    //获得树的深度
    public int getDepth(TreeNode node){
        int maxDepth = 0;
        if (node == null ){
            return maxDepth;
        }
        int left = getDepth(node.left)+1;
        int right  = getDepth(node.right)+1;
        return left > right ? left : right;
    }
//层序遍历 广度优先遍历
    public int maxDepth(TreeNode root) {
        if(root == null) return 0;
        //初始化时放入根节点
        List<TreeNode> queue = new LinkedList<TreeNode>() {
            {
            add(root);
            }
            }, tmp;
        int res = 0;
        while(!queue.isEmpty()) {
            tmp = new LinkedList<>();
            //每次都将下一层的节点放入队列中
            for(TreeNode node : queue) {
                if(node.left != null) tmp.add(node.left);
                if(node.right != null) tmp.add(node.right);
            }
            queue = tmp;
            res++;
        }
        return res;
    }

    //对称二叉树

//层序遍历走不下去啊

    /**
     * 递归的思想，大事化小
     * 终止条件，根节点的左右节点的值一定要相等
     * 左子节点的左子节点和右子节点的右子节点的值相等
     * 左子节点的右子节点和右子节点的左子节点值相等
     * @param root
     * @return
     */
    public boolean isSymmetric(TreeNode root) {
        return root == null ? true : recur(root.left, root.right);
    }
    boolean recur(TreeNode L, TreeNode R) {
        if(L == null && R == null) return true;
        if(L == null || R == null || L.val != R.val) return false;
        return recur(L.left, R.right) && recur(L.right, R.left);
    }
//迭代方式，应用队列，
    public boolean isSymmetric1(TreeNode root) {
        //使用层序遍历
        if(root == null) return true;
        //初始化时放入根节点
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root.left);
        queue.offer(root.right);
        boolean flag =true;
        while(!queue.isEmpty()) {
            TreeNode templ = queue.poll();
            TreeNode tempr = queue.poll();
            if (templ == null && tempr == null) continue;
            if (templ== null || tempr == null||templ.val != tempr.val){
                return false;
            }
            queue.offer(templ.left);
            queue.offer(tempr.right);
            queue.offer(templ.right);
            queue.offer(tempr.left);
        }
        return flag;
    }

    /**
     * 从上到下打印
     * @param root
     * @return
     */
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if(root == null) return result;
        //初始化时放入根节点
        List<TreeNode> queue = new LinkedList<TreeNode>() {
            {
                add(root);
            }
        }, tmp;
        int res = 0;
        List<Integer> a = new ArrayList<>();
        a.add(root.val);
        result.add(a);
        while(!queue.isEmpty()) {
            tmp = new LinkedList<>();
            List<Integer> list = new ArrayList<>();
            //每次都将下一层的节点放入队列中
            for(TreeNode node : queue) {
                if(node.left != null) {
                    tmp.add(node.left);
                    list.add(node.left.val);
                }
                if(node.right != null) {
                    tmp.add(node.right);
                    list.add(node.right.val);
                }
            }
            if (list != null && list.size()>0){
                result.add(list);
            }
            queue = tmp;
        }
        return result;
    }

    private List<List<Integer>> ret;

    public List<List<Integer>> levelOrder1(TreeNode root) {
        ret = new ArrayList<>();
        dfs(0, root);
        return ret;
    }

    private void dfs(int depth, TreeNode root) {
        if (root == null) {
            return;
        }
        if (ret.size() == depth) {
            ret.add(new ArrayList<>());
        }
        ret.get(depth).add(root.val);//
        dfs(depth + 1, root.left);
        dfs(depth + 1, root.right);
    }

    /**
     * 输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)
     *
     * B是A的子结构， 即 A中有出现和B相同的结构和节点值
     * @param args
     */

    public boolean isSubStructure1(TreeNode A, TreeNode B) {


        //约定空树不是任意一个树的子结构
        if(B == null || A == null){
            return false;
        }
        //由顶向下递归,我的想法是先找到A和B相等的节点，再往下递归看树一致不，但是少了如果A中的节点有相同的情况，所以谁想要一定要直接扫描全树

        if (A.left !=null ){
           return isSubStructure(A.left,B);
        }
        if (A.right !=null ){
            return isSubStructure(A.right,B);
        }
        boolean flag =isEqualTwoTree(A,B);

        return flag;
    }
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        return (A != null && B != null) && (isEqualTwoTree(A, B) || isSubStructure(A.left, B) || isSubStructure(A.right, B));

    }
    private boolean isEqualTwoTree(TreeNode A, TreeNode B){
        if(B == null) return true;
        if(A == null || A.val != B.val) return false;
        return isEqualTwoTree(A.left,B.left) && isEqualTwoTree(A.right,B.right);
    }

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {

        if(root == null) return "[]";
        StringBuilder res = new StringBuilder("[");
        Queue<TreeNode> queue = new LinkedList<TreeNode>() {{ add(root); }};
        while(!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if(node != null) {
                res.append(node.val + ",");
                queue.add(node.left);
                queue.add(node.right);
            }
            else res.append("null,");
        }
        res.deleteCharAt(res.length() - 1);
        res.append("]");
        return res.toString();
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if (data.equals("[]")){
            return  null;
        }
        String [] vals = data.substring(1,data.length()-1).split(",");
        TreeNode root = new TreeNode(Integer.parseInt(vals[0]));
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int i = 1;
        while (!queue.isEmpty()){
            TreeNode node = queue.poll();
            if (!vals[i].equals("null")){
                node.left = new TreeNode(Integer.parseInt(vals[i]));
                queue.add(node.left);
            }
            i++;
            if (!vals[i].equals("null")){
                node.right = new TreeNode(Integer.parseInt(vals[i]));
                queue.add(node.right);
            }
            i++;
        }
        return root;
    }

    /**
     * 计算器
     * @param s
     * @return
     */
    public int calculate(String s) {
        //乘法和除法优先级高于加减
        //空格可以跳过
        //利用栈先进后出
        int result = 0;
        char [] temp = s.toCharArray();
        Stack<Integer> stack = new Stack<Integer>();
        for (int i = 0; i < temp.length; i++) {
            if (temp[i]=='*'){
                int c =stack.pop();
                Character c1 = temp[i+1];
                result =result+c*Integer.valueOf(c1);
                stack.push(result);
                i++;
            }else if(temp[i]=='/'){
                int c =stack.pop();
                Character c1 = temp[i+1];
                result =result+(Integer)c/Integer.valueOf(c1);
                stack.push(result);
                i++;
            }else if (temp[i] == ' '){
                continue;
            }else if(temp[i]=='-'){
                //stack.push(temp[i]);
            }
        }
        List<Character> list = new ArrayList<>();

        while (!stack.empty()){
           // Character c = stack.pop();

        }
        return 0;

    }

    /**
     * 给你一个二进制字符串数组 strs 和两个整数 m 和 n 。
     *
     * 请你找出并返回 strs 的最大子集的大小，该子集中 最多 有 m 个 0 和 n 个 1 。
     *
     * 如果 x 的所有元素也是 y 的元素，集合 x 是集合 y 的 子集
     * 状态与选择：状态子集的大小，0和1的个数，选择：多少个0或多少个1
     * 状态转移方程
     *          dp[m][n]= {
     *              dp[]
     *          }
     * @param args
     */
    public int findMaxForm1(String[] strs, int m, int n) {
        int strsNum = strs.length;
        int[][] dp = new int[m + 1][n + 1];

        for (int i = 1; i <= strsNum; i++) {
            int[] cnt = count(strs[i - 1]);
            for (int j = m; j >= 0; j--) {
                for(int k = n; k >= 0; k--) {

                    if (cnt[0] > j || cnt[1] > k) {
                        dp[j][k] = dp[j][k];
                    } else {
                        dp[j][k] = Math.max(dp[j][k], dp[j - cnt[0]][k - cnt[1]] + 1);
                    }
                }
            }
        }

        return dp[m][n];
    }

    // cnt[0] = zeroNums, cnt[1] = oneNums
    public int[] count(String str) {
        int[] cnt = new int[2];
        for (char c : str.toCharArray()) {
            cnt[c - '0']++;
        }
        return cnt;
    }
//如果是有序的数组就可以使用左右指针
    public int[] twoSum1(int [] nums , int b){
        //左右指针
        int [] result = {};
        int left = 0;
        int right = nums.length-1;
        while(left < right){
            if (nums[left]+nums[right]==b){
                result = new int[]{nums[left],nums[right]};
                break;
            }
            else if (nums[left]+nums[right] > b){
                right--;
            }else if (nums[left]+nums[right] < b){
                left++;
            }
        }
        return result;
    }

    /**
     * 完全二叉树节点的个数
     * 先求出二叉树的深度h，然后再记录节点中是null的个数，最后节点数2的h次方减null的个数再减一
     * @param root 时间复杂度是N，有没有可能使用递归方式做
     * @return
     */
    public int countNodes(TreeNode root) {
        if(root == null) return 0;
        //初始化时放入根节点
        List<TreeNode> queue = new LinkedList<TreeNode>() {
            {
                add(root);
            }
        }, tmp;
        int res = 0;
        int nums = 0;
        while(!queue.isEmpty()) {
            tmp = new LinkedList<>();
            //每次都将下一层的节点放入队列中
            for(TreeNode node : queue) {
                if(node.left != null){
                    tmp.add(node.left);
                }
                if(node.right != null) {
                    tmp.add(node.right);
                }
            }
            if(queue.size() != 0){
                nums=queue.size();
            }
            queue = tmp;
            res++;
        }
        return (int) (Math.pow(2,res-1)+nums-1);
    }

    /**
     * BTS（二叉搜索树）：左子节点的值要小于右子节点。对于 BST 的每一个节点node，它的左侧子树和右侧子树都是 BST。
     * BST 的中序遍历结果是有序的（升序）
     * 二分查找法
     * @param root
     * @return  2  2   10
     *                 10
     *移位运算
     */
    public boolean exists(TreeNode root, int level, int k) {
        int bits = 1 << (level - 1);
        TreeNode node = root;
        while (node != null && bits > 0) {
            if ((bits & k) == 0) {
                node = node.left;
            } else {
                node = node.right;
            }
            bits >>= 1;
        }
        return node != null;
    }

    //递归实际
    public int countNodes1(TreeNode root) {
        if (root == null){
            return 0;
        }
        int left = countNodes(root.left);
        int right = countNodes(root.right);
        return left+right+1;
    }
    //一个二叉搜索树，寻找
   // public int kthLargest(TreeNode root, int k){}
    int[][] memo;
    public int longestCommonSubsequence(String text1, String text2) {

        int m = text1.length(), n = text2.length();
        // 备忘录值为 -1 代表未曾计算
        memo = new int[m][n];
        for (int[] row : memo)
            Arrays.fill(row, -1);
            return dp(text1,0,text2,0);
    }

    public int dp(String str1,int i,String str2,int j){
        if(i==str1.length() || j==str2.length()){
            return 0;
        }
        if (memo[i][j] != -1) {
            return memo[i][j];
        }
        //比较每个字符是否相同，对应位置
        if (str1.charAt(i) == str2.charAt(j)){
            memo[i][j]= 1+dp(str1,i+1,str2,j+1);
        }else{
            //那实际有三种可能，可能str1的i位置在最长中，也可能str2中j位置在最长
            memo[i][j] =Math.max(dp(str1,i+1,str2,j),dp(str1,i,str2,j+1));
        }
        return memo[i][j];
    }

    /**
     * 想法是：两个指针i和j
     * base case
     * if(j==nums.size())return 0;
     * if(i==0){
     *  return dp(nums,j,j+1);
     * }
     * nums[i]<nums[j]:1+dp(nums,i+1,j+1)
     * nums[i]>nums[j];dp(nums,i-1,j);
     * nums[i]=nums[j];dp(nums,j,j+1);
     *
     *
     * @param nums
     * @return
     */
    public int lengthOfLIS(int[] nums) {
        int i=nums.length,j=nums.length;
      // base case:j是nums.size
        return dp(nums,0,1);
    }
    int rows=0;

    public int dp(int[] nums,int i,int j){
        if(j==nums.length-1)return rows;
        if(i==-1){
            return dp(nums,j,j+1);
        }
        if (nums[i]<nums[j]){
            rows++;
            return dp(nums,j,j+1);
        }else if ( nums[i]>nums[j]){

            return dp(nums,i-1,j);
        }else {
            return dp(nums,j,j+1);
        }

    }

    /**
     * 利用动态规划+二分查找
     * 利用一个临时数组，然后线性遍历原数组，用二分查找法将新的值插入临时数组中，大的就插入，小的就覆盖
     * @param nums
     * @return
     */
    public int lengthOfLIS1(int[] nums) {
        int[] tails = new int[nums.length];
        int res = 0;
        for(int num : nums) {
            int i = 0, j = res;
            while(i < j) {
                int m = (i + j) / 2;
                if(tails[m] < num) i = m + 1;
                else j = m;
            }
            tails[i] = num;
            if(res == j) res++;
        }
        return res;
    }




    /**
     * 一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内
     * 。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。
     *  二分查找怎么实现
     *  我是不是可以用0
     * @param args
     */
    public int missingNumber(int[] nums) {
        int i=0;
        for(int n:nums){
            if(n!=i){
                break;
            }
            i++;
        }
        return i;
    }
//时间复杂度是n
    //比较中值，如果中值是相等index的，则它前面的没出错，一定是后面的出错了
    //如果不相等，一定是前面的出错了，或者他自己错了
    public int missingNumber1(int[] nums) {
        int i=0; int j = nums.length-1;
        while (i<=j){
            int m = (i+j)/2;
            if (m==nums[m]) i=m+1;
            else j=m-1;
        }
        return i;
    }

    /**
     * 稀疏数组搜索。有个排好序的字符串数组，其中散布着一些空字符串，编写一种方法，找出给定字符串的位置。
     *
     * 示例1:
     *
     *  输入: words = ["at", "", "", "", "ball", "", "", "car", "", "","dad", "", ""], s = "ta"
     *  输出：-1
     *  说明: 不存在返回-1。
     * @param args
     */
    public int findString(String[] words, String s) {
        int i=0;int j=words.length-1;
        int [] res = new int[words.length];
        Arrays.fill(res, -1);
        while (i<=j){
            int m = (i+j)/2;
            while("".equals(words[m])){
                m--;
            }
           // Character[] s = new Character();
            char [] arr = words[m].toCharArray();
            char [] arr1 = s.toCharArray();
            if(res[m] == 1){
                return -1;
            }
            if(words[m].compareTo(s)>0) {
                res[m] = 1;
                j = m - 1;
            }else if(words[m].compareTo(s)>0) {
                i=m+1;
                res[m] = 1;
            }else {
                return m;
            }


        }
        return -1;
    }

    /**
     * 输入：nums1 = [1,2,2,1], nums2 = [2,2]
     * 输出：[2,2]
     * 输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
     * 输出：[4,9]
     * 输出结果中每个元素出现的次数，应与元素在两个数组中出现次数的最小值一致。
     * 我们可以不考虑输出结果的顺序
     * 如果给定的数组已经排好序呢？你将如何优化你的算法？
     * 如果 nums1 的大小比 nums2 小很多，哪种方法更优？
     * 如果 nums2 的元素存储在磁盘上，内存是有限的，并且你不能一次加载所有的元素到内存中，你该怎么办
     * int mid = left + (right - left) / 2;
     * @param args
     */
    public int[] intersect(int[] nums1, int[] nums2) {
            int [] temp = new int[Math.max(nums1.length,nums2.length)];
            int [] res = new int[Math.min(nums1.length,nums2.length)];
            Arrays.fill(temp,-1);
            int h=0;
            for (int i=0;i<nums1.length;i++){
                for (int j=0;j<nums2.length;j++){
                    if (temp[j]==1){
                        continue;
                    }
                    if (nums1[i] == nums2[j]){
                        temp[j]=1;
                        res[h]=nums1[i];
                        h++;
                        break;
                    }
                }
            }
            int [] last = new int[h];
            last = Arrays.copyOf(res,h);
            return last;
    }

    /**
     * 初始时，两个指针分别指向两个数组的头部。每次比较两个指针指向的两个数组中的数字，
     * 如果两个数字不相等，则将指向较小数字的指针右移一位，如果两个数字相等，将该数字添加到答案，
     * 并将两个指针都右移一位。当至少有一个指针超出数组范围时，遍历结束
     * @param nums1
     * @param nums2
     * @return
     */
    public int[] intersect1(int[] nums1, int[] nums2) {
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        int [] temp = new int[Math.max(nums1.length,nums2.length)];
        int [] res = new int[Math.min(nums1.length,nums2.length)];
        Arrays.fill(temp,-1);
        int h=0;
        for (int i=0;i<nums1.length;i++){
            int a=0;int b=nums2.length-1;
            while (a<=b){
                int m = (a+b)/2;
                if(temp[m]==1){
                    continue;
                }
                if(nums2[m] > nums1[i]){
                    b=m-1;
                }else if(nums2[m] < nums1[i]){
                    a=m+1;
                }else{
                    res[h]=nums1[i];
                    h++;
                    temp[m]=1;
                }
            }
        }
        int [] last = new int[h];
        last = Arrays.copyOf(res,h);
        return last;
    }

    /**
     * 给定一个 n x n 矩阵，其中每行和每列元素均按升序排序，找到矩阵中第 k 小的元素。
     * 请注意，它是排序后的第 k 小元素，而不是第 k 个不同的元素。
     * matrix = [
     *    [ 1,  5,  9],
     *    [10, 11, 13],
     *    [12, 13, 15]
     * ],
     * k = 8,
     *
     * 返回 13。
     * @param args
     */
    public int kthSmallest(int[][] matrix, int k) {
        int [] temp = new int[matrix.length*matrix[0].length];
        Set<Integer> set = new HashSet<>();
        int h=0;
        for(int m=0;m<matrix.length;m++){//控制行数
            for(int n=0;n<matrix[m].length;n++){//一行中有多少个元素（即多少列）
                temp[h]=matrix[m][n];
                h++;
            }
        }
        Arrays.sort(temp);
        return temp[k-1];
    }

    /**
     * 给定一个只包含整数的有序数组，每个元素都会出现两次，唯有一个数只会出现一次，找出这个数。
     *
     * 示例 1:
     *
     * 输入: [1,1,2,3,3,4,4,8,8]
     * 输出: 2
     *
     * 示例 2:
     *
     * 输入: [3,3,7,7,10,11,11]
     * 输出: 10
     *
     * 注意: 您的方案应该在 O(log n)时间复杂度和 O(1)空间复杂度中运行。
     * 因为有两个值是一样的，所以当我们拿走中间一样的这两个数时，左右两个数组的肯定有一个是奇数
     * 所以肯定在奇数的那里
     *
     * @param args
     */
    public int singleNonDuplicate(int[] nums) {
        int left=0;int right=nums.length-1;
        if (nums.length==1){
            return nums[0];
        }
        //注意如果left=right，说明只剩下一个数了，肯定就是要的那个，因为要比较mid++，所以这个right边界应该要小-1
        while (left<right){
            int mid = left + (right-left)/2;
            if(nums[mid] == nums[mid+1]){
                int len = right-mid+1;
                if(len%2==0){
                    right = mid-1;
                }else{
                    left = mid+2;
                }
            }else if (nums[mid] == nums[mid-1]){
                int len = right-mid;
                if(len%2==0){
                    right = mid-2;
                }else{
                    left = mid+1;
                }
            }else if (nums[mid] != nums[mid+1]&&nums[mid] != nums[mid-1]){
                return nums[mid];
            }


        }
        //别越界
        return -1;
    }

    /**  未解决。可以使用二分查找+哈希表
     * 也可以是滑动窗口
     * 动态规划
     *  输入：
     * A: [1,2,3,2,1]
     * B: [3,2,1,4,7]
     * 输出：3
     * 解释：
     * 长度最长的公共子数组是 [3, 2, 1] 。
     * @param args
     */
    public int findLength(int[] A, int[] B) {
        Arrays.sort(B);
        int []res = new int[B.length];
        Arrays.fill(res,-1);
        int h=0;
        for(int i=0;i<A.length;i++){
            int left = 0;int right=B.length-1;
            while (left<=right){
                int mid = left+(right-left)/2;
                if(B[mid]==A[i]){
                    if(res[mid]!=1){
                        h++;
                        res[mid]=1;
                        break;
                    }else {

                    }
                }else if(B[mid]>A[i]){
                    right = mid-1;
                }else if(B[mid]<A[i]){
                    left = mid+1;
                }


            }
        }

        return h;
    }

    /**
     * 输入: citations = [0,1,3,5,6]
     * 输出: 3
     * 解释: 给定数组表示研究者总共有 5 篇论文，每篇论文相应的被引用了 0, 1, 3, 5, 6 次。
     *      由于研究者有 3 篇论文每篇至少被引用了 3 次，其余两篇论文每篇被引用不多于 3 次，所以她的 h 指数是 3。
     * @param args
     *
     */
    public int hIndex(int[] citations) {

            int left=0;int right = citations.length-1;
            while (left<=right){
                int mid = left + (right-left)/2;
                if(citations[mid] == citations.length-mid){
                    return citations[mid];
                }else if(citations[mid] > citations.length-mid){
                    right = mid-1;
                }else{
                    left =mid +1;
                }
            }
            return citations.length-left;//-1
    }

    /**
     * 一个 N x N 的坐标方格 grid 中，每一个方格的值 grid[i][j] 表示在位置 (i,j) 的平台高度。
     *
     * 现在开始下雨了。当时间为 t 时，此时雨水导致水池中任意位置的水位为 t 。
     * 你可以从一个平台游向四周相邻的任意一个平台，
     * 但是前提是此时水位必须同时淹没这两个平台。假定你可以瞬间移动无限距离，也就是默认在方格内部游动是不耗时的。
     * 当然，在你游泳的时候你必须待在坐标方格里面。
     *
     * 你从坐标方格的左上平台 (0，0) 出发。最少耗时多久你才能到达坐标方格的右下平台 (N-1, N-1)？
     *
     * 输入: [[0,2],[1,3]]
     * 输出: 3
     * 解释:
     * 时间为0时，你位于坐标方格的位置为 (0, 0)。
     * 此时你不能游向任意方向，因为四个相邻方向平台的高度都大于当前时间为 0 时的水位。
     *
     * 等时间到达 3 时，你才可以游向平台 (1, 1). 因为此时的水位是 3，坐标方格中的平台没有比水位 3 更高的，所以你可以游向坐标方格中的任意位置
     *
     * @param args
     */

    /**
     * 给定一个二叉树的根节点 root ，返回它的 中序 遍历。左中右
     * @param args
     */
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        midErgodic(root,list);
        return list;
    }
    private void midErgodic(TreeNode x,List<Integer>  keys){
        if (x==null){
            return;
        }
        //先递归，把左子树中的键放到keys中
        if (x.left!=null){
            midErgodic(x.left,keys);
        }
        //把当前结点x的键放到keys中
        keys.add(x.val);
        //在递归，把右子树中的键放到keys中
        if(x.right!=null){
            midErgodic(x.right,keys);
        }

    }

    class Node {
        public int val;
        public Node left;
        public Node right;
        public Node next;

        public Node() {}

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, Node _left, Node _right, Node _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    };
    /**
     * 给定一个 N 叉树，返回其节点值的前序遍历 中左右
     * @param args
     */
    /*public List<Integer> preorder(Node root) {
        //层序遍历+栈先进后出
        List<Integer> res = new ArrayList<>();
        LinkedList<Node> stack = new LinkedList<>();
        if (root == null) {
            return res;
        }
        stack.add(root);
        while (!stack.isEmpty()){
            // 因为先进要后出，所以反转下nodes这样，在后面的总是先出
            Node tempNode = stack.pollLast();
            List<Node> nodes =tempNode.children;
            Collections.reverse(nodes);
            for(Node node : nodes){
                stack.add(node);
            }
            res.add(tempNode.val);
        }
        return res;
    }*/

    /**
     * 最大树定义：一个树，其中每个节点的值都大于其子树中的任何其他值。
     *
     * 给出最大树的根节点 root。
     *
     * 就像之前的问题那样，给定的树是从表 A（root = Construct(A)）递归地使用下述 Construct(A) 例程构造的：
     *
     *     如果 A 为空，返回 null
     *     否则，令 A[i] 作为 A 的最大元素。创建一个值为 A[i] 的根节点 root
     *     root 的左子树将被构建为 Construct([A[0], A[1], ..., A[i-1]])
     *     root 的右子树将被构建为 Construct([A[i+1], A[i+2], ..., A[A.length - 1]])
     *     返回 root
     *
     * 请注意，我们没有直接给定 A，只有一个根节点 root = Construct(A).
     *
     * 假设 B 是 A 的副本，并附加值 val。保证 B 中的值是不同的。
     *
     * 返回 Construct(B)。
     * @param args
     */

    /**
     * 给定二叉树根结点 root ，此外树的每个结点的值要么是 0，要么是 1。
     *
     * 返回移除了所有不包含 1 的子树的原二叉树。
     *
     * ( 节点 X 的子树为 X 本身，以及所有 X 的后代。)
     *
     * 示例1:
     * 输入: [1,null,0,0,1]
     * 输出: [1,null,0,null,1]
     *
     * 解释:
     * 只有红色节点满足条件“所有不包含 1 的子树”。
     * 右图为返回的答案。
     *
     * @param args
     */
    public TreeNode pruneTree(TreeNode root) {
        return containsOne(root)?root:null;
    }

    private boolean containsOne(TreeNode root){
        if (root == null) return false;
        boolean a = containsOne(root.left);
        boolean b = containsOne(root.right);
        if(!a){
            root.left = null;
        }
        if(!b){
            root.right = null;
        }
        return root.val == 1 || a ||b;
    }

    /**
     * 给定一棵二叉树，设计一个算法，创建含有某一深度上所有节点的链表
     * （比如，若一棵树的深度为 D，则会创建出 D 个链表）。返回一个包含所有深度的链表的数组
     * @param args
     */
    /*public ListNode[] listOfDepth(TreeNode tree) {
        //采用层序遍历
        Queue<TreeNode> queue = new LinkedList<>();

        ListNode [] listNodes = new ListNode[]{};

        return null;
    }*/
    public class ListNode {
        int val;
        ListNode next;
        ListNode(int x) { val = x; }
   }

    /**
     *给定一个字符串S，检查是否能重新排布其中的字母，使得两相邻的字符不同。
     *
     * 若可行，输出任意可行的结果。若不可行，返回空字符串。
     *
     * 示例 1:
     *
     * 输入: S = "aab"
     * 输出: "aba"
     *
     * 示例 2:
     *
     * 输入: S = "aaab"
     * 输出: ""
     * @param args
     */
        //贪心算法。交替放置最常见的字母
        public String reorganizeString(String S) {
            if (S.length() < 2) {
                return S;
            }
            int[] counts = new int[26];
            int maxCount = 0;
            int length = S.length();
            for (int i = 0; i < length; i++) {
                char c = S.charAt(i);
                counts[c - 'a']++;
                maxCount = Math.max(maxCount, counts[c - 'a']);
            }
            if (maxCount > (length + 1) / 2) {
                return "";
            }
            char[] reorganizeArray = new char[length];
            int evenIndex = 0, oddIndex = 1;
            int halfLength = length / 2;
            for (int i = 0; i < 26; i++) {
                char c = (char) ('a' + i);
                while (counts[i] > 0 && counts[i] <= halfLength && oddIndex < length) {
                    reorganizeArray[oddIndex] = c;
                    counts[i]--;
                    oddIndex += 2;
                }
                while (counts[i] > 0) {
                    reorganizeArray[evenIndex] = c;
                    counts[i]--;
                    evenIndex += 2;
                }
            }
            return new String(reorganizeArray);
        }

    /**
     *   给定一个链表，判断链表中是否有环。
     * 如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，我们使用整数 pos
     * 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，
     * 仅仅是为了标识链表的实际情况。
     * 如果链表中存在环，则返回 true 。 否则，返回 false 。
     *
     * @param args
     */
    public boolean hasCycle(ListNode head) {
        ListNode slow,fast;
        slow = fast =head;
        while (fast!=null && fast.next != null){

            slow =head.next;
            fast = head.next.next;
            if(slow == fast){
                return true;
            }
        }

        return false;
    }

    /**
     * 给定一个链表和一个特定值 x，对链表进行分隔，使得所有小于 x 的节点都在大于或等于 x 的节点之前。
     * 你应当保留两个分区中每个节点的初始相对位置。
     * @param args
     */
    public ListNode partition(ListNode head, int x) {
        //两个指针
        ListNode before_head = new ListNode(0);
        ListNode before = before_head;
        ListNode after_head = new ListNode(0);
        ListNode after = after_head;

        while(head != null){
            if (head.val<x){
                before.next = head;
                before = before.next;
            }else{
                after.next = head;
                after = after.next;
            }
            head=head.next;
        }
        after.next = null;
        before.next=after_head.next;
        return before_head.next;
    }

    /**
     * 给定一个只包含 '(' 和 ')' 的字符串，找出最长的包含有效括号的子串的长度
     * @param args
     */

    /**
     * 给你一个按升序排序的整数数组 num（可能包含重复数字），请你将它们分割成一个或多个子序列，
     * 其中每个子序列都由连续整数组成且长度至少为 3 。
     * 如果可以完成上述分割，则返回 true ；否则，返回 false 。
     * @param args
     */
    public boolean isPossible(int[] nums) {
        //长度大于等于3 nums至少大于3
        if(nums.length<3){
            return false;
        }
        return true;
    }

    /**
     * 给你一个树，请你 按中序遍历 重新排列树，使树中最左边的结点现在是树的根，并且每个结点没有左子结点，只有一个右子结点
     * 构建树要使用Integer，使用构造方法
     * 中序遍历如果是一个平衡树，中序后则有序
     * @param args
     */
    public TreeNode increasingBST(TreeNode root) {
        //中序遍历---左中右
        Queue<Integer> keys = new LinkedList<>();
        midErgodic(root,keys);
        TreeNode node=new TreeNode(keys.poll());
        TreeNode node1=node;
        for (int i =keys.size();i>0;i--){
            if(i==0){
                node.right = null;
            }
            node.right =new TreeNode(keys.poll());
            node.left=null;
            node = node.right;
        }
        return node1;
    }
    private void midErgodic(TreeNode x,Queue<Integer> keys){
        if (x == null){
            return;
        }
        if (x.left != null){
            midErgodic(x.left,keys);
        }
        keys.add(x.val);
        if (x.right != null){
            midErgodic(x.right,keys);
        }
    }

    /**
     *  给定一棵二叉树，设计一个算法，创建含有某一深度上所有节点的链表
     *  （比如，若一棵树的深度为 D，则会创建出 D 个链表）。返回一个包含所有深度的链表的数组。
     *  把每层都打印一遍
     */
    public ListNode[] listOfDepth(TreeNode tree) {
        ListNode[] list = new ListNode[getDepth1(tree)];
        LinkedList<TreeNode> quece = new LinkedList<>();
        quece.add(tree);
        while (!quece.isEmpty()){
            LinkedList<TreeNode> temp = new LinkedList<>();
            TreeNode node = quece.poll();
        }
        return null;
    }
    public int getDepth1(TreeNode node){
        int i=1;
        if(node==null){
            return i;
        }
        int left =getDepth(node.left)+1;
        int right =getDepth(node.right)+1;
        return Math.max(left,right);
    }

    /**
     * 构建一个树
     * @param nums
     * @return
     */
    public static TreeNode constructTree(Integer[] nums) {
        if (nums.length == 0) return new TreeNode(0);
        Deque<TreeNode> nodeQueue = new LinkedList<>();
        // 创建一个根节点
        TreeNode root = new TreeNode(nums[0]);
        nodeQueue.offer(root);
        TreeNode cur;
        // 记录当前行节点的数量（注意不一定是2的幂，而是上一行中非空节点的数量乘2）
        int lineNodeNum = 2;
        // 记录当前行中数字在数组中的开始位置
        int startIndex = 1;
        // 记录数组中剩余的元素的数量
        int restLength = nums.length - 1;

        while (restLength > 0) {
            // 只有最后一行可以不满，其余行必须是满的
//            // 若输入的数组的数量是错误的，直接跳出程序
//            if (restLength < lineNodeNum) {
//                System.out.println("Wrong Input!");
//                return new TreeNode(0);
//            }
            for (int i = startIndex; i < startIndex + lineNodeNum; i = i + 2) {
                // 说明已经将nums中的数字用完，此时应停止遍历，并可以直接返回root
                if (i == nums.length) return root;
                cur = nodeQueue.poll();
                if (nums[i] != null) {
                    cur.left = new TreeNode(nums[i]);
                    nodeQueue.offer(cur.left);
                }
                // 同上，说明已经将nums中的数字用完，此时应停止遍历，并可以直接返回root
                if (i + 1 == nums.length) return root;
                if (nums[i + 1] != null) {
                    cur.right = new TreeNode(nums[i + 1]);
                    nodeQueue.offer(cur.right);
                }
            }
            startIndex += lineNodeNum;
            restLength -= lineNodeNum;
            lineNodeNum = nodeQueue.size() * 2;
        }

        return root;
    }

    /**
     * 快速排序
     * @param args
     */
    public void quickSort(Comparable [] target){
        Comparable flag = target[0];

    }
    //先分组
    private int partition(Comparable [] a,int lo,int hi){
        //确定分界值
        Comparable key = a[lo];
        //定义两个指针，分别指向待切分元素的最小索引处和最大索引处的下一个位置
        int left=lo;
        int right=hi+1;

        //切分
        while(true){
            //先从右往左扫描，移动right指针，找到一个比分界值小的元素，停止
            while(less(key,a[--right])){
                if (right==lo){
                    break;
                }
            }

            //再从左往右扫描，移动left指针，找到一个比分界值大的元素，停止
            while(less(a[++left],key)){
                if (left==hi){
                    break;
                }
            }
            //判断 left>=right,如果是，则证明元素扫描完毕，结束循环，如果不是，则交换元素即可
            if (left>=right){
                break;
            }else{
                exch(a,left,right);
            }
        }

        //交换分界值
        exch(a,lo,right);

        return right;
    }
    private static boolean less(Comparable v, Comparable w) {
        return v.compareTo(w) < 0;
    }
    private static void exch(Comparable[] a, int i, int j) {
        Comparable t = a[i];
        a[i] = a[j];
        a[j] = t;
    }

    /**
     *  给定一种规律 pattern 和一个字符串 str ，判断 str 是否遵循相同的规律。
     * 这里的 遵循 指完全匹配，例如， pattern 里的每个字母和字符串 str 中的每个非空单词之间存在着双向连接的对应规律。
     */
    public boolean wordPattern(String pattern, String s) {
        String []arr = s.split(" ");
        if(pattern.length()!=arr.length){
            return false;
        }

        int i = 0;
        Map<Character,String> map = new HashMap<>();
        while (i<arr.length){
            char A = pattern.charAt(i);

            if(!map.containsKey(A)){
                if(!map.containsValue(arr[i])){
                    map.put(A,arr[i]);
                    continue;
                }else {
                    return false;
                }
            }

            String temp = map.get(A);
            if(temp==null||!temp.equals(arr[i])){
                return false;
            }
            i++;
        }
        return true;
    }

    /**
     * 给你一个二进制字符串数组 strs 和两个整数 m 和 n
     * 请你找出并返回 strs 的最大子集的大小，该子集中 最多 有 m 个 0 和 n 个 1 。
     *
     * 如果 x 的所有元素也是 y 的元素，集合 x 是集合 y 的 子集 。
     *
     * @param args
     */
    public int findMaxForm(String[] strs, int m, int n) {
        return 0;
    }

    /**
     * 实现 int sqrt(int x) 函数。
     * 计算并返回 x 的平方根，其中 x 是非负整数。
     * 由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。
     * @param args
     */
    public int mySqrt(int x) {
        int l = 0, r = x, ans = -1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if ((long) mid * mid <= x) {
                ans = mid;
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return ans;
    }

    /**
     * 给定一个二维平面及平面上的 N 个点列表Points，其中第i个点的坐标为Points[i]=[Xi,Yi]。
     * 请找出一条直线，其通过的点的数目最多。
     * 设穿过最多点的直线所穿过的全部点编号从小到大排序的列表为S，你仅需返回[S[0],S[1]]作为答案，
     * 若有多条直线穿过了相同数量的点，则选择S[0]值较小的直线返回，S[0]相同则选择S[1]值较小的直线返回。
     * @param args
     */
    public int[] bestLine(int[][] points) {

        return null;
    }

    /**
     * 给你两个字符串 s 和 t ，请你通过若干次以下操作将字符串 s 转化成字符串 t ：
     *     选择 s 中一个 非空 子字符串并将它包含的字符就地 升序 排序。
     * 比方说，对下划线所示的子字符串进行操作可以由 "14234" 得到 "12344" 。
     * 如果可以将字符串 s 变成 t ，返回 true 。否则，返回 false 。
     * 一个 子字符串 定义为一个字符串中连续的若干字符
     * @param args
     */
    public boolean isTransformable(String s, String t) {
        return true;
    }

    /**
     * 「快乐前缀」是在原字符串中既是 非空 前缀也是后缀（不包括原字符串自身）的字符串。
     *
     * 给你一个字符串 s，请你返回它的 最长快乐前缀。
     *
     * 如果不存在满足题意的前缀，则返回一个空字符串
     * @param args
     */
    public String longestPrefix(String s) {
        int left=0;
        int right=s.length();
        return null;

    }

    /**
     * 给定两个字符串 s 和 t，它们只包含小写字母。
     *
     * 字符串 t 由字符串 s 随机重排，然后在随机位置添加一个字母。
     *
     * 请找出在 t 中被添加的字母。
     * @param args
     */
    public char findTheDifference(String s, String t) {
        int ret = 0;
        for (int i = 0; i < s.length(); ++i) {
            ret ^= s.charAt(i);
            System.out.println((char)ret);
        }
        for (int i = 0; i < t.length(); ++i) {
            ret ^= t.charAt(i);
            System.out.println((char)ret);
        }
        return (char) ret;
    }

    /**
     * 数组的每个索引作为一个阶梯，第 i个阶梯对应着一个非负数的体力花费值 cost[i](索引从0开始)。
     *
     * 每当你爬上一个阶梯你都要花费对应的体力花费值，然后你可以选择继续爬一个阶梯或者爬两个阶梯。
     *
     * 您需要找到达到楼层顶部的最低花费。在开始时，你可以选择从索引为 0 或 1 的元素作为初始阶梯
     * @param args
     */
    public int minCostClimbingStairs(int[] cost) {
        //从cost[0]开始，选择走一步，也可以走两步，但是应该是按照小的走
        //从cost[1]开始，选择走一步，也可以走两步，但是应该按照小的走
        //dp数组中表示的是最小，因为选择是可以走2步也可以1步，所以dp[i]应该是其中最小的一个，
        int n = cost.length;
        int[] dp = new int[n + 1];
        dp[0] = dp[1] = 0;
        for (int i = 2; i <= n; i++) {
            dp[i] = Math.min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2]);
        }
        return dp[n];
        //滚动数组，实际就是临时变量，每次赋值
        /**
         * int n = cost.length;
         *         int prev = 0, curr = 0;
         *         for (int i = 2; i <= n; i++) {
         *             int next = Math.min(curr + cost[i - 1], prev + cost[i - 2]);
         *             prev = curr;
         *             curr = next;
         *         }
         *         return curr;
         *
         */
    }

    /**
     * 几乎每一个人都用 乘法表。但是你能在乘法表中快速找到第k小的数字吗？
     * 给定高度m 、宽度n 的一张 m * n的乘法表，以及正整数k，你需要返回表中第k 小的数字。
     *
     * @param args
     */
    public int findKthNumber(int m, int n, int k) {
        //1,1处肯定是最小的，然后行和竖肯定是其次的，也就是带1，带2的
      //  1	2	3
     //   2	4	6
     //   3	6	9
        //  m/2肯定是   [1.m]和[n,1]
        return 0;
    }

    public boolean leafSimilar(TreeNode root1, TreeNode root2) {
        //深度优先遍历
        List<Integer> list1 = new ArrayList<>();
        List<Integer> list2 = new ArrayList<>();
        dfs(root1,list1);
        dfs(root2,list2);
        return list1.equals(list2);
    }

    private void dfs(TreeNode root,List<Integer> list){
        if(root.left==null && root.right==null){
            list.add(root.val);
        }
        if(root.left != null)dfs(root.left,list);
        if(root.right != null)dfs(root.right,list);
    }

    /**
     * 班上有 N 名学生。其中有些人是朋友，有些则不是。他们的友谊具有是传递性。如果已知 A 是 B 的朋友，B 是 C 的朋友，
     * 那么我们可以认为 A 也是 C 的朋友。所谓的朋友圈，是指所有朋友的集合。
     *
     * 给定一个 N * N 的矩阵 M，表示班级中学生之间的朋友关系。如果M[i][j] = 1，表示已知第 i 个和 j 个学生互为朋友关系，
     * 否则为不知道。你必须输出所有学生中的已知的朋友圈总数。
     *
     * @param args
     */

    /**
     * 给定一组 N 人（编号为 1, 2, ..., N）， 我们想把每个人分进任意大小的两组。
     *
     * 每个人都可能不喜欢其他人，那么他们不应该属于同一组。
     *
     * 形式上，如果 dislikes[i] = [a, b]，表示不允许将编号为 a 和 b 的人归入同一组。
     *
     * 当可以用这种方法将所有人分进两组时，返回 true；否则返回 false。
     * @param args
     */
    public boolean possibleBipartition(int N, int[][] dislikes) {
        return false;
    }

    /**
     * 数组中占比超过一半的元素称之为主要元素。给定一个整数数组，找到它的主要元素。若没有，返回-1。
     * @param args
     */
    public int majorityElement(int[] nums) {
        int mid = nums.length/2;
        Map<Integer,Integer> map = new HashMap<>();
        for (int i=0;i<nums.length;i++){
            map.put(nums[i],map.get(nums[i])!=null ? map.get(nums[i])+1:1);
            if(map.get(nums[i])!=null&&map.get(nums[i])>mid){
                return nums[i];
            }
        }
        return -1;
    }
    //摩尔投票
    public int majorityElement1(int[] nums) {
        //维护一个众数和一个频数
        int major=0;
        int vote=0;
        for(int num:nums){
            if(vote==0){
                major=num;
                vote++;
            }else{
                if(major==num){
                    vote++;
                }else{
                    vote--;
                }
            }
        }
        if(vote==0){
            return -1;
        }
        int identify=0;
        for (int num:nums){
            if (num == major) {
                identify++;
                if (identify > nums.length / 2) {
                    return major;
                }
            }
        }
        return -1;
    }

    /**
     * 给定一个按非递减顺序排序的整数数组 A，返回每个数字的平方组成的新数组，要求也按非递减顺序排序。
     * @param args  O(n)
     */
    public int[] sortedSquares(int[] nums) {
        /*int [] result = new int[nums.length];
        for(int i=0;i<nums.length;i++){
            result[i]=nums[i]*nums[i];
        }
        Arrays.sort(result);
        return result;*/
        //双指针，因为已经排好序的数组，如果都是正数，平方后也是，如果
        //1.找到正负的分界出全是负数则分界值应该是nums.length,全是正数，则分界值为-1
        int partion=-1;
        for(int i=nums.length-1;i>0;i--){
            if(nums[i]<=0){
                partion=i;
                break;
            }
        }
        int [] arr = new int[nums.length];
        int left = partion;
        int right = partion+1;
        int index=0;
        while (left>=0&&right<nums.length){
            int left2=nums[left]*nums[left];
            int right2=nums[right]*nums[right];
            if(left<0){
                arr[index]=right2;
                right++;
            }else if(right==nums.length){
                arr[index]=left2;
                left--;
            }
            else if(left2>right2){
                arr[index]=right2;
                right++;
            }else if(left2<=right2){
                arr[index]=left2;
                left--;
            }
            index++;
        }
        if(left<0){
            while (right<nums.length){
                arr[index]=nums[right]*nums[right];
                right++;
                index++;
            }
        }
        if(right>=nums.length){
            while (left>=0){
                arr[index]=nums[left]*nums[left];
                left--;
                index++;
            }
        }
        return arr;
    }

    /**
     * 给你一个由 不同 整数组成的整数数组 arr 和一个整数 k 。
     * 每回合游戏都在数组的前两个元素（即 arr[0] 和 arr[1] ）之间进行。比较 arr[0] 与 arr[1] 的大小，
     * 较大的整数将会取得这一回合的胜利并保留在位置 0 ，较小的整数移至数组的末尾。当一个整数赢得 k 个连续回合时，游戏结束，该整数就是比赛的 赢家 。
     * 返回赢得比赛的整数。
     * 题目数据 保证 游戏存在赢家。
     * @param args
     */
    public int getWinner(int[] arr, int k) {
        int frequency =0;

        while (frequency!=k){
            //比较0和1的大小
            int []temp = new int[arr.length];
            if(arr[0]>arr[1]){
                //交换1和末尾的
                temp[0]=arr[0];
                temp[temp.length-1]=arr[1];
                frequency++;
            }else{
                temp[0]=arr[1];
                temp[temp.length-1]=arr[0];
                frequency=1;
            }
            System.arraycopy(arr,2,temp,1,temp.length-2);
            arr = temp;
            if(frequency==arr.length){
                break;
            }
        }
        return arr[0];
    }
    private void change(int i,int j,int []arr){
        int temp = arr[i];
        arr[i]=arr[j];
        arr[j]=temp;
    }

    /**
     * 给你一个整数数组 arr，请你判断数组中是否存在连续三个元素都是奇数的情况：如果存在，请返回 true ；否则，返回 false 。
     *
     * @param args
     */
    public boolean threeConsecutiveOdds(int[] arr) {
        for(int i=0;i<arr.length;i++){
            if(arr[i]%2==0){
                continue;
            }
            if(i+2<arr.length){
                if((arr[i+1]&1)!=0&(arr[i+2]&1)!=0){//判断奇数偶数
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * 给定一个二叉树，返回其节点值的锯齿形层序遍历。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）
     * @param args
     */
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if(root==null){
            return result;
        }
        Stack<TreeNode> stack = new Stack<>();
        Stack<TreeNode> temp=null;
        boolean flag = true;
        stack.add(root);
        while (stack.size()>0){
            temp = new Stack<>();
            List<Integer> list = new ArrayList<>();
            int length=stack.size();
            if(flag){
                //从左到右
                flag = false;
                for(int i=0;i<length;i++){
                    TreeNode node = stack.pop();
                    list.add(node.val);
                    if(node.left!=null)temp.add(node.left);
                    if(node.right!=null)temp.add(node.right);
                }

            }else{
                //从右到左
                flag = true;
                for(int i=0;i<length;i++){
                    TreeNode node = stack.pop();
                    list.add(node.val);
                    if(node.right!=null)temp.add(node.right);
                    if(node.left!=null)temp.add(node.left);
                }
            }
            result.add(list);
            stack.clear();
            stack = temp;
        }
        return result;
    }

    /**
     * 给你一个字符串 s 和一个 长度相同 的整数数组 indices 。
     *
     * 请你重新排列字符串 s ，其中第 i 个字符需要移动到 indices[i] 指示的位置。
     *
     * 返回重新排列后的字符串
     * @param args
     */
    public String restoreString(String s, int[] indices) {
        Map<Integer,Character> map = new HashMap<>();
        for (int i=0;i<indices.length;i++){
            map.put(indices[i],s.charAt(i));
        }
        StringBuilder sb = new StringBuilder();
        for (int i=0;i<indices.length;i++){
            sb.append(map.get(i));
        }
        return sb.toString();
    }

    /**
     * 给你两个数组，arr1 和 arr2，
     *
     *     arr2 中的元素各不相同
     *     arr2 中的每个元素都出现在 arr1 中
     *
     * 对 arr1 中的元素进行排序，使 arr1 中项的相对顺序和 arr2 中的相对顺序相同。未在 arr2 中出现过的元素需要按照升序放在 arr1 的末尾。
     * @param args
     */
    public int[] relativeSortArray(int[] arr1, int[] arr2) {
        return null;
    }

    /**
     * 给定一个字符串，找到它的第一个不重复的字符，并返回它的索引。如果不存在，则返回 -1。
     * @param args
     */
    public int firstUniqChar(String s) {
        Map<Character,Integer> map = new HashMap<>();
        Map<Character,Integer> map1 = new HashMap<>();
        Character [] ch = new Character[s.length()];
       for(int i=0;i<s.length();i++){
           if(!map1.containsKey(s.charAt(i))){
               if(map.containsKey(s.charAt(i))){
                   map1.put(s.charAt(i),i);
                   map.remove(s.charAt(i));
               }else{
                   map.put(s.charAt(i),i);
               }
           }

       }
       int min=-1;
       for (int i:map.values()){
           if(min==-1){
               min = i;
           }
            if(min>i){
                min = i;
            }
       }
       return min;
    }

    /**
     * 假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。
     *
     * 对每个孩子 i，都有一个胃口值 g[i]，这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 j，
     * 都有一个尺寸 s[j] 。如果 s[j] >= g[i]，我们可以将这个饼干 j 分配给孩子 i ，这个孩子会得到满足。
     * 你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。
     * @param args
     */
    public int findContentChildren(int[] g, int[] s) {
        //最小的吃最小的。最大的吃最大的
        Arrays.sort(g);
        Arrays.sort(s);
        int numOfChildren = g.length, numOfCookies = s.length;
        int count = 0;
        for (int i = 0, j = 0; i < numOfChildren && j < numOfCookies; i++, j++) {
            while (j < numOfCookies && g[i] > s[j]) {
                j++;
            }
            if (j < numOfCookies) {
                count++;
            }
        }
        return count;
    }
    //给定一个二叉树，原地将它展开为一个单链表
    public void flatten(TreeNode root) {
        //前序遍历，中左右
        Queue<TreeNode> queue = new LinkedList<>();
        beforesort(root,queue);
        int length = queue.size();
        for(int i=0;i<length;i++){
            TreeNode node = queue.poll();
            ListNode lnode = new ListNode(node.val);

        }

    }
    private void beforesort(TreeNode node,Queue<TreeNode> queue){
        if (node.left != null) beforesort(node.left,queue);
        queue.add(node.left);
        queue.add(node);
        if(node.right != null) beforesort(node.right,queue);
        queue.add(node.right);
    }

    /**
     * 给定一个已按照升序排列 的有序数组，找到两个数使得它们相加之和等于目标数。
     *
     * 函数应该返回这两个下标值 index1 和 index2，其中 index1 必须小于 index2。
     *
     * 说明:
     *
     *     返回的下标值（index1 和 index2）不是从零开始的。
     *     你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。
     * @param args
     */
    public int[] twoSum13(int[] numbers, int target) {
        int left=0;int right=numbers.length-1;
        while (left<right){
            int temp = numbers[left]+numbers[right];
            if(temp==target){
                return new int[]{left+1,right+1};
            }else if (temp > target){
                right--;
            }else{
                left++;
            }
        }
        return new int[]{};
    }

    /**
     * 假设你有一个很长的花坛，一部分地块种植了花，另一部分却没有。可是花卉不能种植在相邻的地块上，它们会争夺水源，两者都会死去。
     *
     * 给定一个花坛（表示为一个数组包含0和1，其中0表示没种植花，1表示种植了花），和一个数 n 。能否在不打破种植规则的情况下种入 n 朵花？
     * 能则返回True，不能则返回False。。
     * @param args
     */
    public boolean canPlaceFlowers(int[] flowerbed, int n) {
        for (int i=0;i<flowerbed.length;i++){
            if(flowerbed[i]==0&&n!=0){
                if(i+1<flowerbed.length&&flowerbed[i+1]==0||i==flowerbed.length-1){
                    if ((i-1>0&&flowerbed[i-1]==0)||i==0){
                        flowerbed[i]=1;
                        n--;
                    }
                }
            }
        }
        boolean flag=false;
        if(n==0){
            flag=true;
        }
        return flag;
    }

    /**
     * 给你一个字符串 s 和一个整数数组 cost ，其中 cost[i] 是从 s 中删除字符 i 的代价。
     *
     * 返回使字符串任意相邻两个字母不相同的最小删除成本。
     *
     * 请注意，删除一个字符后，删除其他字符的成本不会改变。
     * @param args
     */
    public int minCost(String s, int[] cost) {
        int speed = 0;
        if(s.length()<=1){
            return speed;
        }
        for (int i=0,j=1;i<s.length()&&j<s.length();){
            char a = s.charAt(i);
            //我只删除小的哪一个
            if(j<s.length()&&a==s.charAt(j)){
                if (cost[i]<cost[j]){
                    speed=speed+cost[i];
                    i=j;
                    j=j+1;
                }else{
                    speed=speed+cost[j];
                    j=j+1;
                }
            }else {
                i=j;
                j=j+1;
            }
        }
        return speed;
    }

    /**
     * 编写一个函数来查找字符串数组中的最长公共前缀。
     * 如果不存在公共前缀，返回空字符串 ""
     * @param args
     */
    public String longestCommonPrefix(String[] strs) {
        String str="";
        if(strs.length==1){
            return strs[0];
        }
        for(int i=1;i<=strs.length-1;i++){
            String str1="";
            String temp = strs[i-1];
            String temp1 = strs[i];
            int min = Math.min(temp.length(),temp1.length());
            for (int j =0;j<min;j++){
                if(temp.charAt(j)==temp1.charAt(j)){
                    str1 = str1+temp.charAt(j);
                }else{
                    break;
                }
            }
            if (str1.length()<str.length()||i==1){
                str = str1;
            }
            if ("".equals(str1)){
                str="";
                break;
            }
        }
        return str;
    }

    public String longestCommonPrefix1(String[] strs) {
        String str = "";
        if(strs.length==0){
            return "";
        }
        str = strs[0];
        for(String a:strs){
            if("".equals(str)){
                return "";
            }
            while (!a.startsWith(str)){
                str = str.substring(0,str.length()-1);
            }
        }
        return str;
    }

    /**
     * 编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：
     *
     *     每行中的整数从左到右按升序排列。
     *     每行的第一个整数大于前一行的最后一个整数
     * @param args
     */
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length-1;
        int n =0;
        if(matrix.length==0){
            return false;
        }
        n = matrix[0].length-1;

        int left =0;int right = m;
        int left1= 0;int right1=n;
        while(left<right||right==0){
            if(left1>=right1){
                break;
            }
            while (left1<right1){
                if(left>right){
                    break;
                }
                int mid = left + (right-left)/2;
                int mid1 = left1+(right1-left1)/2;
                int temp = matrix[mid][mid1];
                if(temp==target){
                    return true;
                }else if(temp<target){
                    if(matrix[mid][n]>target){
                        left1=mid1+1;
                    }else if(matrix[mid][n]<target){
                        left=mid+1;
                        if(right==0){
                            return false;
                        }
                        break;

                    }else{
                        return true;
                    }

                }else if (temp>target){
                    if(matrix[mid][0]>target){
                        right=mid-1;
                        if(right==0){
                            return false;
                        }
                        break;
                    }else if(matrix[mid][0]<target){
                        right1= mid1-1;
                    }else{
                        return true;
                    }

                }
            }
        }
        if(matrix[0][0]==target){
            return true;
        }
        return false;

    }

    /**
     * 给定两个字符串 s 和 t，判断它们是否是同构的。
     *
     * 如果 s 中的字符可以被替换得到 t ，那么这两个字符串是同构的。
     *
     * 所有出现的字符都必须用另一个字符替换，同时保留字符的顺序。两个字符不能映射到同一个字符上，但字符可以映射自己本身。
     * @param args
     */
    public boolean isIsomorphic(String s, String t) {
        String temp = "";
        if(s.length()!=t.length()){
            return false;
        }
        Map<Character,Character> map = new HashMap<>();
        for(int i=0;i<s.length();i++){
            char s1 = s.charAt(i);
            char t1 = t.charAt(i);
            if(!map.containsKey(s1)){
                if(!map.containsValue(t1)){
                    temp = temp+ t1;
                    map.put(s1,t1);
                }else{
                    return false;
                }
            }else{
                temp = temp+ map.get(s1);
            }
        }
        if(temp.equals(t)){
            return true;
        }
        return false;
    }

    /**
     * 每一回合，从中选出两块 最重的 石头，然后将它们一起粉碎。假设石头的重量分别为 x 和 y，且 x <= y。那么粉碎的可能结果如下：
     *
     *     如果 x == y，那么两块石头都会被完全粉碎；
     *     如果 x != y，那么重量为 x 的石头将会完全粉碎，而重量为 y 的石头新重量为 y-x。
     *
     * 最后，最多只会剩下一块石头。返回此石头的重量。如果没有石头剩下，就返回 0。
     * @param args
     * Simulate the process. We can do it with a heap, or by sorting some list of stones every time we take a turn.
     */
    public int lastStoneWeight(int[] stones) {
        if(stones.length==1){
            return stones[0];
        }
     //   int [] heap = new int[stones.length+1];
     //   heap[0]=0;
        //最大堆
      //  System.arraycopy(stones,0,heap,1,stones.length);
        genHeap(stones);
        while (stones[2]!=0){
            int temp= Math.abs(stones[0]-stones[1]);
            if(temp==0){
                stones[0]=0;
                stones[1]=0;
            }else{
                stones[0]=0;
                stones[1]=temp;
            }
            genHeap(stones);
        }
        return stones[0];
    }
    //手写最大堆
    public int[] creatMaxHeap(int [] heap){
        //利用堆的特性 顶点是k，则堆的左子节点是2k，右子节点是2k+1
        //下沉算法
        siftDown(1,heap);
        return heap;
    }
    public void siftDown(Integer index,int []data){
        int N=data.length-1;
        while (index*2<=N){
            int maxIndex;
            if (index*2+1>N){
                maxIndex = index*2;
            }else{
                //找到左右子节点的值最大的一个
                maxIndex = data[index*2]<data[index*2+1]?index*2+1:index*2;
            }
            if (data[index]>(data[maxIndex])){
                break;
            }
            //交换最大索引处的值和当前索引
            int temp = data[index];
            data[index] = data[maxIndex];
            data[maxIndex]=temp;
            //
            index = maxIndex;
        }

    }

    // 生成堆
    private void genHeap(int[] nums){
        for(int i = (nums.length - 2)/2; i >= 0; i--){
            change(nums, i);
        }
    }

    // 调整
    private void change(int[] nums, int start){
        int next = 2 * start + 1;
        int tmp = nums[start];
        while(next < nums.length){
            if(next < nums.length-1 && nums[next+1] > nums[next])
                next++;
            if(nums[next] > tmp){
                nums[start] = nums[next];
                start = next;
                next = 2 * start + 1;
            }else{
                break;
            }
        }
        nums[start] = tmp;
    }

    /**
     * 给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有满足条件且不重复的三元组。
     *
     * 注意：答案中不可以包含重复的三元组。
     * @param args
     */
    public List<List<Integer>> threeSum(int[] nums) {
        //双指针
        Arrays.sort(nums);
        List<List<Integer>> list = new LinkedList<>();
        for (int i = 0; i < nums.length-2; i++) {
            if((i==0 ) || (i>0 && nums[i]!= nums[i-1])){
                int target = 0-nums[i];
                //然后再剩下的数组里面找到两数之和等于target的
                int left = i+1;
                int right = nums.length-1;
                while (left < right){
                   // List<Integer> arrayList = new ArrayList<>();
                    int sum = nums[left]+nums[right];
                    if (target == sum){
                        //Arrays.asList(nums[i], nums[left], nums[right])
                        //Arrays.asList()可以'将数组转化成list但是不是ArrayList，想转成arrayList，new ArrayList(Arrays.asList(arr))
                        list.add(Arrays.asList(nums[i], nums[left], nums[right]));
                        //元素相同要后移，防止加入重复的 list
                        while (left < right && nums[left] == nums[left+1]) left++;
                        while (left < right && nums[left] == nums[right-1]) right--;
                        left++;
                        right--;
                    }else if (target > sum){
                        left++;
                    }else {
                        right--;
                    }
                }
            }
        }

        return list;
    }

    /**
     * 斐波那契数，通常用 F(n) 表示，形成的序列称为 斐波那契数列 。该数列由 0 和 1 开始，后面的每一项数字都是前面两项数字的和。也就是：
     *
     * F(0) = 0，F(1) = 1
     * F(n) = F(n - 1) + F(n - 2)，其中 n > 1
     *
     * 给你 n ，请计算 F(n) 。
     *
     * @param args
     */
    public int fib(int n) {
        if(n<=1){
            return n;
        }
        int [] temp = new int [n+1];
        temp[0]=0;temp[1]=1;
        for(int i=1;i<n;i++){
            temp[i+1]=temp[i]+temp[i-1];
        }
        return temp[n];
    }

    /**
     *
     * 在一个由小写字母构成的字符串 s 中，包含由一些连续的相同字符所构成的分组。
     *
     * 例如，在字符串 s = "abbxxxxzyy" 中，就含有 "a", "bb", "xxxx", "z" 和 "yy" 这样的一些分组。
     *
     * 分组可以用区间 [start, end] 表示，其中 start 和 end 分别表示该分组的起始和终止位置的下标。上例中的 "xxxx" 分组用区间表示为 [3,6] 。
     *
     * 我们称所有包含大于或等于三个连续字符的分组为 较大分组 。
     *
     * 找到每一个 较大分组 的区间，按起始位置下标递增顺序排序后，返回结果。
     *
     * @param args
     */
    public List<List<Integer>> largeGroupPositions(String s) {
        if(s==null || "".equals(s)||s.length()<=2){
            return new ArrayList<>();
        }
        //双指针方法
        List<List<Integer>> res = new ArrayList<>();
        int i=0;
        int j=1;
        while(j<s.length()&&i<s.length()){
           char temp = s.charAt(i);
           char temp1 = s.charAt(j);
           if(temp==temp1){
               if(j==s.length()-1&&j-i>=2){
                   List<Integer> list = new ArrayList<>();
                   list.add(i);
                   list.add(j);
                   res.add(list);
                   break;
               }
               j++;

           }else {
               if(j-i>=3){
                   List<Integer> list = new ArrayList<>();
                   list.add(i);
                   list.add(j-1);
                   res.add(list);
               }
               i=j;
               j++;
           }
        }

        return res;
    }

    /**
     * 给你一个由一些多米诺骨牌组成的列表 dominoes。
     *
     * 如果其中某一张多米诺骨牌可以通过旋转 0 度或 180 度得到另一张多米诺骨牌，我们就认为这两张牌是等价的。
     *
     * 形式上，dominoes[i] = [a, b] 和 dominoes[j] = [c, d] 等价的前提是 a==c 且 b==d，
     * 或是 a==d 且 b==c。
     *
     * 在 0 <= i < j < dominoes.length 的前提下，找出满足 dominoes[i] 和 dominoes[j]
     * 等价的骨牌对 (i, j) 的数量。
     * @param args
     */
    public int numEquivDominoPairs(int[][] dominoes) {
        if(dominoes.length<=0){
            return 0;
        }
        Set<String> set = new HashSet<>();
        for(int m=0;m<dominoes.length;m++){//控制行数
            String zh = "";
            for(int n=0;n<dominoes[m].length;n++){//一行中有多少个元素（即多少列）
                zh +=String.valueOf(dominoes[m][n]);
            }
            if(set.contains(zh)){
                set.remove(zh);
            }else{
                if(set.contains(reverse3(zh))){
                    set.remove(reverse3(zh));
                }else {
                    set.add(zh);
                  //  set.add(reverse3(zh));
                }
            }
        }
        return dominoes.length - set.size();

    }
    public static String reverse3(String s) {
        char[] array = s.toCharArray();
        String reverse = "";  //新建空字符串
        for (int i = array.length - 1; i >= 0; i--)
            reverse += array[i];
        return reverse;

    }

    /**
     * 可重复的数组，奇数放在奇数位，偶数放在偶数位
     * @param args
     */
    public int[] oddNumAndEven(int [] nums){
        int i = 0;
        int j = 1;
        List<Integer> odd = new ArrayList<>();
        List<Integer> even = new ArrayList<>();
        while (i<nums.length-1&&j<nums.length-1){
            if (nums[i]%2!=0){
                odd.add(i);
            }
            if (nums[j]%2==0){
                even.add(j);
            }
            i = i+2;
            j = j+2;
        }
        for (int o=0 ;o<Math.max(odd.size(),even.size());o++){
            int temp = nums[odd.get(o)];
            nums[odd.get(o)] = nums[even.get(o)];
            nums[even.get(o)] = temp;
        }
        return nums;
    }
/**========================================剑指offer===============================================*/
    /**
     * 在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。
     * 请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数
     * @param target
     * @param array
     * @return
     */
    public boolean Find(int target, int [][] array) {
        int n = array.length;
        int m = array[0].length-1;//列
        for(int i=0;i<n;i++){
            int left = 0;
            int right = m;
            while(left<=right){
                int mid = left + (right-left)/2;
                if (array[i][mid]==target){
                    return true;
                }else if (array[i][mid]<target){
                    left = mid+1;
                }else {
                    right = mid-1;
                }
            }
        }
        return false;
    }

    /**
     * 请实现一个函数，将一个字符串中的每个空格替换成“%20”。
     * 例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。
     * @param args
     */
    public String replaceSpace(String s) {
        s = s.replace(" ","%20");
        return s;
    }

    /**
     * 输入一个链表，按链表从尾到头的顺序返回一个ArrayList。
     * @param args
     */
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        //栈，先进后出Stack
        if (listNode == null){
            return new ArrayList<>();
        }
        Stack<ListNode> stack = new Stack<>();
        stack.push(listNode);
        while (listNode.next!=null){
            stack.push(listNode.next);
            //别忘记重新赋值
            listNode = listNode.next;
        }
        ArrayList<Integer> arrayList = new ArrayList<>();
        while (stack!=null && stack.size()>0){
            ListNode temp =stack.pop();
            arrayList.add(temp.val);
        }
        return arrayList;
    }

    /**
     * 输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。
     * 假设输入的前序遍历和中序遍历的结果中都不含重复的数字
     * 。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
     *
     * 前序遍历：中左右，  中序遍历：左中右
     * @param pre
     * @param in
     * @return
     */
    public TreeNode root_node;
    public TreeNode reConstructBinaryTree(int [] pre,int [] in) {
         //前序遍历的第一个肯定是根节点,第二个肯定是左子树的根节点
        //中序遍历的根节点往左为左子树，左子树的数组的长度，在前序遍历加2就是右子树的根节点
        //采用递归和分枝
        if(pre==null|| in==null||pre.length==0||in.length==0)return null;
        int root = pre[0];
        TreeNode rootNode = new TreeNode(root);
        root_node =rootNode;
        int index_root = 0;
        for (int i=0 ;i<in.length;i++){
            if (root == in[i]){
                index_root = i;
                break;
            }
        }
        if(in != null &&  index_root !=0){
            int [] leftTree_in = new int[index_root];
            int [] rightTree_in = new int[index_root];
            int [] leftTree_pre = new int[index_root];
            int [] rightTree_pre = new int[index_root];
            System.arraycopy(in,0,leftTree_in,0,index_root);
            System.arraycopy(in,index_root+1,rightTree_in,0,in.length-index_root-1);

            System.arraycopy(pre,1,leftTree_pre,0,index_root);
            System.arraycopy(pre,index_root+1,rightTree_pre,0,index_root);
            rootNode.left = reConstructBinaryTree(leftTree_pre,leftTree_in);
            rootNode.right = reConstructBinaryTree(rightTree_pre,rightTree_in);
        }
        return root_node;
    }

    Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();

    public void push(int node) {
        //维护一个栈，在每次进栈前，都将栈1全部移动到另一个栈，然后再回来
        if(stack1==null||stack1.size()==0){
            stack1.push(node);
            return;
        }
        int size_1 = stack1.size();
        for (int i=0;i<size_1;i++){
            int temp = stack1.pop();
            stack2.push(temp);
        }
        stack1.push(node);
        int size2=stack2.size();
        for (int i=0;i<size2;i++){
            int temp = stack2.pop();
            stack1.push(temp);
        }


    }

    public int pop() {
       return stack1.pop();
    }

    /**
     * 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
     * 输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
     * NOTE：给出的所有元素都大于0，若数组大小为0，请返回0
     * @param array
     * @return
     */
    public int minNumberInRotateArray(int [] array) {
        if (array.length == 0){
            return 0;
        }
        //TODO:
        int right=array.length-1;
        int left=0;
        while(left<right){
            int mid = left+(right-left)/2;
            //左边界大于中间值，561223
            System.out.println(array[left]);
            System.out.println(array[mid]);
            if (array[left]>array[mid]){
                left = left+1;
                //左边界小于中间值  345612
            }else if(array[left]<array[mid]){

                right = mid-1;


            }else{//44456或者444443
                return array[left];
            }
        }
        return array[left];
    }

    /**
     * 大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0，第1项是1）。
     *F(0)=0，F(1)=1, F(n)=F(n - 1)+F(n - 2)（n ≥ 2，n ∈ N*）
     * n≤39n\leq 39n≤39
     * @param args
     */
    Map<Integer,Integer> map = new HashMap<Integer,Integer>();
    public int Fibonacci(int n) {
        if(n==0){
            return 0;
        }
        if (n==1){
            return 1;
        }

        map.put(0,0);
        map.put(1,1);
        Integer f1 = map.get(n-1);
        if (f1==null){
            f1 = Fibonacci(n-1);
            map.put(n-1,f1);
        }
        Integer f2 = map.get(n-2);
        if (f2==null){
            f2 = Fibonacci(n-2);
            map.put(n-2,f2);
        }
        return (int)f1+(int)f2;
    }

    /**
     * 一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。
     * @param args
     */
    public int JumpFloor(int target) {
        if(target == 1){
            return 1;
        }
        if (target == 2){
            return 2;
        }
        return JumpFloor(target-2)+JumpFloor(target-1);

    }

    /**
     * 一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
     * @param args
     */
    public int jumpFloorII(int target) {
        if(target == 1){
            return 1;
        }
        if (target == 2){
            return 2;
        }
        return 0;
    }

    /**
     *我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？
     * @param args
     */
    public int rectCover(int target) {
    return 0;
    }

    /**输入一个整数，输出该数32位二进制表示中1的个数。其中负数用补码表示
     * 负整数的补码，将其原码除符号位外的所有位取反（0变1，1变0，符号位为1不变）后加1
     * @param args
     */
    public int NumberOf1(int n) {
        //区分n是正负整数
        if(n==0)return 0;
        if (n>0){
            //十进制转化成二进制
            String a = Integer.toBinaryString(n);
            char [] arr = a.toCharArray();
            int alength = a.length();
            int result = 0;
            for (int i = 0;i<arr.length;i++){
                if(arr[i]=='1'){
                    result++;
                }
            }
            return result;
        }else{
            String a = Integer.toBinaryString(n);
            char [] arr = a.toCharArray();
            int alength = a.length();
            int result = 0;
            for (int i = 0;i<arr.length;i++){
                if(arr[i]=='1'){
                    result++;
                }
            }

            return result;
        }
    }

    /**
     * 给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
     * 保证base和exponent不同时为0
     * @param args
     */
    public double Power(double base, int exponent) {
        if (exponent==0){
            return 1;
        }
        if (base==0.0){
            return 0;
        }
        if(exponent>0){
            int i =1;
            double base1=1;
            while (i<=exponent){
                base1 = base1*base;
                i++;
            }
            return base1;
        }else{
            //a的负n次方 等于a的n次方分之一
            int exponent_abs = Math.abs(exponent);
            int i =1;
            double base1=1;
            while (i<=exponent_abs){
                base1 = base1*base;
                i++;
            }
            base1 = 1/base1;
            return base1;
        }


    }

    /**
     * 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，
     * 所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。
     * @param args
     */
    public int[] reOrderArray (int[] array) {
        Queue<Integer> list_j = new LinkedList<>();
        for (int i=0;i<array.length;i++){
            if(array[i]%2==1){
                list_j.add(array[i]);
            }
        }
        Queue<Integer> list_o = new LinkedList<>();
        for (int i=0;i<array.length;i++){
            if(array[i]%2==0){
                list_o.add(array[i]);
            }
        }
        int [] result = new int[array.length];
        int length_j = list_j.size();
        for (int i= 0;i<length_j;i++){
            result[i] =list_j.poll();
        }
        int length_o = list_o.size();
        for (int i= length_j;i<length_j+length_o;i++){
            result[i] =list_o.poll();
        }
        return result;
    }

    /**
     * 输入一个链表，输出该链表中倒数第k个结点。
     * @param args
     */
    public ListNode FindKthToTail (ListNode pHead, int k) {
        // 双指针方法：指针
        if(pHead == null || k==0){
            return null;
        }
        int i=1;
        ListNode pk= pHead;
        for (;i<k;i++){
            if(pk.next == null){
                return null;
            }
            pk = pk.next;
        }

        while (pk.next !=null){
            pk = pk.next;
            pHead = pHead.next;
        }
        return pHead;
    }

    /**
     * 输入一个链表，反转链表后，输出新链表的表头。
     * @param args
     */
    public ListNode ReverseList(ListNode head) {
        if(head == null){
            return null;
        }
        //借助栈stack
        Stack<Integer> stack = new Stack<Integer>();
        stack.push(head.val);
        while (head.next != null){
            stack.push(head.next.val);
            head = head.next;
        }

        int a = stack.pop();
        ListNode root = new ListNode(a);
        ListNode temp = root;
        int size = stack.size();
        for (int i =0;i<size;i++){
            int j = stack.pop();
            temp.next = new ListNode(j);
            temp = temp.next;
        }
        return root;
    }

    /**
     * 输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则
     * @param args
     */
    public ListNode Merge(ListNode list1,ListNode list2) {
        if (list1 == null && list2 == null){
            return null;
        }
        if (list1 == null){
            return list2;
        }
        if (list2 == null){
            return list1;
        }
        //1的头结点小于2的头结点，则1的头结点是整个头结点
        if(list1.val <= list2.val){
            list1.next = Merge(list1.next,list2);
            return list1;
        }else{
            list2.next = Merge(list1,list2.next);
            return list2;
        }

    }

    /**
     * 输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）
     * @param args
     */
    public boolean HasSubtree(TreeNode root1,TreeNode root2) {
        if (root1 == null || root2 == null){
            return false;
        }
        //想法是遍历整个A，然后和B的根一致的就判断左右子树是否是一样的
        if (root1 == root2){
            return isSubTree(root1.left,root2.left)&&isSubTree(root1.right,root2.left);
        }

        return HasSubtree(root1.left,root2)||HasSubtree(root1.right,root2);


    }

    private boolean isSubTree(TreeNode root1,TreeNode root2){
        if (root1 == null || root2 == null){
            return true;
        }
        if (root1 == null ){
            return false;
        }
        if (root2 == null){
            return false;
        }
        if (root1.val != root2.val){
            return false;
        }else{
            return isSubTree(root1.left,root2.left)&&isSubTree(root1.right,root2.left);
        }

    }

    /**
     * 操作给定的二叉树，将其变换为源二叉树的镜像
     * @param args
     */
    public TreeNode Mirror (TreeNode pRoot) {
        //层序打印+栈
        Stack1<TreeNode> stack = new Stack1<>();
        Stack1<TreeNode> tempstack = null;
        Stack1<TreeNode> tempstack1 = null;
        Stack1<TreeNode> tempstack2 = null;
        stack.push(pRoot);

        while (!stack.isEmpty()){
            tempstack = new Stack1<>();
            tempstack1 = new Stack1<>();
            tempstack2 = new Stack1<>();
            for (TreeNode node: stack){
                if (node.left != null) tempstack.add(node.left);
                if (node.right != null) tempstack.add(node.right);
            }
            tempstack1 = (Stack1<TreeNode>) stack.clone();
            stack = (Stack1<TreeNode>) tempstack.clone();
            int tem_size = tempstack.size();
            TreeNode temp =null;
            for (int i= 0;i<tem_size;i++){
                if (i%2==0){
                    temp = tempstack1.pop();
                    temp.left=tempstack.pop();
                }else{
                    temp.right=tempstack.pop();
                }

            }
        }
        return pRoot;
    }
    class Stack1<T> extends Stack<T> implements Cloneable{
        public Stack<T> stack = new Stack<T>();
        @Override
        public synchronized Stack<T> clone() {
            int size = this.size();
            for (int i=0;i<size;i++){
                stack.add(this.pop());
            }
            Stack<T> stack1 = new Stack<T>();
            int size1 = stack1.size();
            for (int i=0;i<size;i++){
                stack1.add(stack.pop());
            }
            return stack1;
        }
    }

    /**
     * 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。
     * 例如序列1,2,3,4,5是某栈的压入顺序，
     * 序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）
     * @param args
     */
    public boolean IsPopOrder(int [] pushA,int [] popA) {
        Stack<Integer> stack = new Stack<>();
        int len = pushA.length;
        for (int i=0,j=0;i<len;i++){
            if (pushA[i]!= popA[j]){
                stack.push(pushA[i]);
            }else{
                j++;
                while (!stack.isEmpty() && stack.peek() == popA[j]){
                    stack.pop();
                    j++;
                }

            }
        }
        return stack.isEmpty();

    }

    /**
     * 从上往下打印出二叉树的每个节点，同层节点从左至右打印。
     * @param args
     */
    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.add(root);
        ArrayList<Integer> arrayList = new ArrayList<>();
        while (!queue.isEmpty()){
            Queue<TreeNode> queue_temp = new LinkedList<TreeNode>();
            int size = queue.size();
            for (int i=0;i<size;i++){
                TreeNode temp = queue.poll();
                if (temp.left!=null){
                    queue_temp.add(temp.left);
                }

                if (temp.right!=null){queue_temp.add(temp.right);}
                arrayList.add(temp.val);
            }
            queue = queue_temp;
        }
        return arrayList;
    }

    /**
     *
     * @param args
     */
    public int lengthOfLongestSubstring(String s) {
        int n = s.length();
        Set<Character> set = new HashSet<>();
        int ans = 0, i = 0, j = 0;
        while (i < n && j < n) {
            //如果有重复的j就先不动，等这i动，慢慢的从前移
            //移动窗口
            if (!set.contains(s.charAt(j))){
                set.add(s.charAt(j++));
                ans = Math.max(ans, j - i);
            }
            else {
                set.remove(s.charAt(i++));
            }
        }
        return ans;
    }

    public int lengthOfLongestSubstring1(String s) {
        //采用跳跃式跟新i
        /*int n = s.length();
        Map<Character,Integer> map = new HashMap<>();
        int ans = 0;
        for (int j = 0, i = 0; j < n; j++) {
            if (map.containsKey(s.charAt(j))) {
                i = Math.max(map.get(s.charAt(j)), i);
            }
            ans = Math.max(ans, j - i + 1);
            map.put(s.charAt(j), j + 1);//下标 + 1 代表 i 要移动的下个位置
        }
        return ans;*/

        int n = s.length(), ans = 0;
        int[] index = new int[128];
        for (int j = 0, i = 0; j < n; j++) {
            i = Math.max(index[s.charAt(j)], i);
            ans = Math.max(ans, j - i + 1);
            index[s.charAt(j)] = j + 1;//（下标 + 1） 代表 i 要移动的下个位置
        }
        return ans;
    }

    /**
     * 判断是否是
     * @param args
     */
    public boolean VerifySquenceOfBST(int [] sequence,int start,int root) {
        if(start >= root){
            return true;
        }
        //先找左右子树的分割节点
        int separation = start;
        for (;separation<root;separation++){
            if(sequence[separation]>sequence[root]){

                break;
            }
        }

        //在右子树查看是否有小于root的结点

        for (int i = separation; i < root; i++) {
            if (sequence[i] < sequence[root]) {
                return false;
            }
        }

        return VerifySquenceOfBST(sequence,start,separation-1)&&VerifySquenceOfBST(sequence,separation,root-1);

    }

    /**
     * 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如
     * ，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
     * 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10
     * @param args
     */
    public ArrayList<Integer> printMatrix(int [][] matrix) {
        return null;
    }

    /**
     * 输入一颗二叉树的根节点和一个整数，按字典序打印出二叉树中结点值的和为输入整数的所有路径。
     * 路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。
     *
     * 1.成为叶子节点的条件，改节点的左右子节点都是null
     * 2.先序遍历，根节点，左子节点，右子节点
     * @param args
     */
    public ArrayList<ArrayList<Integer>> res_arrayLst = new ArrayList<>();
    ArrayList<Integer> records = new ArrayList<>();
    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root,int target) {
        if (root == null){
            return res_arrayLst;
        }

        records.add(root.val);
        target = target - root.val;
        //符合条件的是叶子节点和target为0才会是结果
        if (root.left == null && root.right == null && target == 0){
            res_arrayLst.add(new ArrayList<>(records));
        }
        ArrayList<ArrayList<Integer>> result1 = FindPath(root.left, target);
        ArrayList<ArrayList<Integer>> result2 = FindPath(root.right, target);
        records.remove(records.size()-1);
        return res_arrayLst;
        /*if(root == null)return result;
        list.add(root.val);
        target -= root.val;
        if(target == 0 && root.left == null && root.right == null)
            result.add(new ArrayList<Integer>(list));
//因为在每一次的递归中，我们使用的是相同的result引用，所以其实左右子树递归得到的结果我们不需要关心，
//可以简写为FindPath(root.left, target)；FindPath(root.right, target)；
//但是为了大家能够看清楚递归的真相，此处我还是把递归的形式给大家展现了出来。
        ArrayList<ArrayList<Integer>> result1 = FindPath(root.left, target);
        ArrayList<ArrayList<Integer>> result2 = FindPath(root.right, target);
        list.remove(list.size()-1);
        return result;*/
    }

    /**
     * 输入一个复杂链表（每个节点中有节点值，以及两个指针，
     * 一个指向下一个节点，另一个特殊指针random指向一个随机节点），请对此链表进行深拷贝，
     * 并返回拷贝后的头结点。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）
     * @param args
     */
    /**
     * 输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。
     * @param args
     */
    TreeNode pre=null;

    public TreeNode Convert(TreeNode pRootOfTree) {
        //双向链表
        //递归，每次的根结点的左指针指向左子结点的右子节点，右指针指向右子节点的左子节点
        //记录原来的左右子节点
        TreeNode root=null;
        if (pRootOfTree==null)
            return null;
        Convert(pRootOfTree.left);
        if (root==null){
            root=pRootOfTree;
        }
        if (pre!= null){
            pRootOfTree.left=pre;
            pre.right=pRootOfTree;
        }
        pre=pRootOfTree;
        Convert(pRootOfTree.right);
        return root;


    }

    /**105.
     * 利用中序遍历和前序遍历恢复二叉树
     * @param args
     */
    public TreeNode buildTree(int [] In_order ,int [] pre_order){
        //通过前序遍历获得根节点

        return helper(In_order, 0, In_order.length, pre_order, 0, pre_order.length);
    }
    private TreeNode helper(int [] In_order ,int in_start,int in_end,int [] pre_order,int pre_start,int pre_end){
        if (In_order.length == 0|| pre_order.length == 0){
            return null;
        }
        TreeNode root = new TreeNode(pre_order[pre_start]);
        int mid = 0;
        for (int i = 0; i < In_order.length; i++) {
            if (In_order[i] == pre_order[0]){
                mid = i;break;
            }
        }
        int len = mid-in_start;
        root.left =helper(In_order,in_start,mid-1,pre_order,pre_start+1,pre_start+1+len);
        root.right = helper(In_order,mid+1,in_end,pre_order,pre_start+1+len+1,pre_end);
        return root;
    }

    /**
     * 给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。
     * 暴力循环法，时间复杂度n3
     * @param args
     */
    public double findMedianSortedArrays1(int[] nums1, int[] nums2) {
        //将两个数组合并，然后采用归并排序，判断新数组的个数
        //奇数就是（n+1）/2，偶数计算是（n/2-1和n/2）/2
        //归并排序
        //TODO：
        return 0.0;
    }

    /**给你一个字符串 s，找到 s 中最长的回文子串
     *
     */
    public String longestPalindrome(String s) {
        //然后找到所有的字串
        String res = "";
        for (int i = 0; i < s.length() ; i++) {
            for (int j = i+1; j <= s.length(); j++) {
                String temp = s.substring(i, j);
                if (isPalindromic(temp)) {
                    if (temp.length() > res.length()) {
                        res = new String(temp);
                    }
                }
            }
        }
        return res;
    }

    /**
     * 判断字符串是否是回文串
     * @param args
     */
    public boolean isPalindromic(String s){
        for (int i = 0 ;i<s.length()/2;i++){
            if (s.charAt(i) != s.charAt(s.length()-1-i)){
                return false;
            }
        }
        return true;
    }

    /**
     * longestPalindrome最长回文串
     * 改变：求反转字符串和字符串的最长公共字串。‘abcbajjk’和‘kjjabcba’
     * 思路，把俩个字串当作一个二维数组的行和列
     * @param args
     */
    public String longestPalindrome1(String s){
        if ("".equals(s)){
            return "";
        }
        String s1 = new StringBuffer(s).reverse().toString();
        char[] chars = s.toCharArray();
        char[] chars1 = s1.toCharArray();
        int[][] pchar = new int[s.length()][s.length()];
        int maxLen = 0;
        int maxEnd = 0;
        for (int i = 0; i < s.length(); i++) {
            for (int j = 0; j < s.length(); j++) {
                if (chars[i] == chars1[j]) {
                    if (i == 0 || j == 0) {
                        pchar[i][j] = 1;
                    } else {
                        pchar[i][j] = pchar[i - 1][j - 1] + 1;
                    }
                }


                /**********修改的地方*******************/
                if (pchar[i][j] > maxLen) {
                    int beforeRev = s.length() - 1 - j;
                    if (beforeRev + pchar[i][j] - 1 == i) { //判断下标是否对应
                        maxLen = pchar[i][j];
                        maxEnd = i;
                    }
                    /*************************************/
                }

            }
        }
        return s.substring(maxEnd - maxLen + 1, maxEnd + 1);
    }

    /**Z字变换
     * 将一个给定字符串 s 根据给定的行数 numRows ，以从上往下、从左到右进行 Z 字形排列。
     *
     * 比如输入字符串为 "PAYPALISHIRING" 行数为 3 时，排列如下：
     *
     * P   A   H   N
     * A P L S I I G
     * Y   I   R
     *
     * 之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如："PAHNAPLSIIGYIR"。
     *
     * 请你实现这个将字符串进行指定行数变换的函数：
     *
     * string convert(string s, int numRows);
     * @param args
     */
    public String convert(String s, int numRows){
        //二维数组的第一个参数是行，第二个是列
        char [][] p = new char[numRows][s.length()];
        int j=0;
        int x=0;
        boolean b = false;//false:j++但是不换列，true：j--，换列
        for (int i = 0; i < s.length() ; i++) {
            //j是行，x是列
            //如果j是numRows-1，换列，x+1  j--，直到j=0是才不换列 j++
            p[j][x]= s.charAt(i);
            if (!b){
                if (j==numRows-1){
                    x++;
                    if(j>0) j--;
                    b = true;
                }else{
                    j++;
                }
            }else{
                //因为numRows==1根本不可能会换行
                if (j==0&&numRows>1){
                    j++;
                    b=false;
                }else{
                    x++;
                    if(j>0) j--;
                }
            }
        }
        String s1 = "";
        for (int i = 0; i <numRows ; i++) {
            for (int k = 0; k < p[0].length; k++) {
                if(p[i][k] != '\u0000'){
                    s1+=p[i][k];
                }
            }
        }
        return s1;
        //需要优化现在时间复杂度是平方，空间复杂度是n平方
    }

    public String convert1(String s, int numRows){
        //规律是一个周期是cycleLen，然后按照周期遍历就好了
        //比如：第一，他的下一位的是第二个周期的第一个,也就是0+cycleLen，然后判断0+cycleLen是否越界，没有的话
        //就能使用，第二个参数
        if (numRows == 1)
            return s;

        StringBuilder ret = new StringBuilder();
        int n = s.length();
        int cycleLen = 2 * numRows - 2;

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j + i < n; j += cycleLen) { //每次加一个周期
                ret.append(s.charAt(j + i));
                if (i != 0 && i != numRows - 1 && j + cycleLen - i < n) //除去第 0 行和最后一行
                    ret.append(s.charAt(j + cycleLen - i));
            }
        }
        return ret.toString();
    }

    /**
     * 整形反转
     * @param x
     * @return
     */
    public int reverse(int x) {
        int rev = 0;
        while (x != 0) {
            int pop = x % 10;
            x /= 10;
            if (rev > Integer.MAX_VALUE/10 ) return 0;
            if (rev < Integer.MIN_VALUE/10 ) return 0;
            rev = rev * 10 + pop;
        }
        return rev;
    }

    public int recursion(int n){
        if (n == 0 || n==1){
            return 1;
        }
        return n * recursion(n-1);
    }

    /**
     * 最能盛水的容器，
     * @param args
     */
    public int maxArea(int[] height) {
        //暴力递归
        int capacity = 0;
        for (int i = 0; i <height.length ; i++) {
            for (int j = i+1; j < height.length; j++) {
                capacity = Math.max(Math.max(height[i],height[j])*j-i,capacity);
            }
        }
        return capacity;
    }
    public int maxArea2(int[] height) {
        //先从最外侧的两个柱子算起，然后找出相较短的，因为容量是有短的决定，所以移动短的，才能更新
        //有点双指针的意思，遍历一次就ok，因为下标的长度有可可能是相等的，所以高度的变话最终要
        //比如说，第2和倒数第3根，之间的宽度和倒数第2根与第3根的宽度一样
        int maxarea = 0, l = 0, r = height.length - 1;
        while (l < r) {
            maxarea = Math.max(maxarea, Math.min(height[l], height[r]) * (r - l));
            if (height[l] < height[r])
                l++;
            else
                r--;
        }
        return maxarea;
    }


        public int maxArea3(int[] height) {
            // int n = height.length;
            int left = 0;
            int leftH = height[left];
            int right = height.length - 1;
            int rightH = height[right];
            int sum = Math.min(leftH,rightH)*(right-left);
            while (left < right) {
                if (leftH <= rightH){
                    left++;
                    if (height[left] > leftH){
                        leftH = height[left];
                        if (Math.min(height[left], height[right])*(right - left) > sum) {
                            sum = Math.min(height[left], height[right])*(right - left);
                        }
                    }
                }else {
                    right--;
                    if (height[right] > rightH){
                        rightH = height[right];
                        if (Math.min(height[left], height[right])*(right - left) > sum) {
                            sum = Math.min(height[left], height[right])*(right - left);
                        }
                    }
                }
            }

            return sum;
        }

    /**
     * 罗马数字包含以下七种字符： I， V， X， L，C，D 和 M。
     *
     * 字符          数值
     * I             1
     * V             5
     * X             10
     * L             50
     * C             100
     * D             500
     * M             1000
     *
     * 例如， 罗马数字 2 写做 II ，即为两个并列的 1。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。
     *
     * 通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：
     *
     *     I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。
     *     X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。
     *     C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。
     *
     * 给你一个整数，将其转为罗马数字。
     * @param args
     */
    public String intToRoman(int num) {
        Map<Integer,String> map = new HashMap<>();
        map.put(4,"IV");
        map.put(9,"IX");
        map.put(40,"XL");
        map.put(90,"XC");
        map.put(400,"CD");
        map.put(900,"CM");
        String res = map.get(num);
        if (res != null &&!"".equals(res)){
            return res;
        }
        int Mnum = num / 1000 ;
        num = num-(Mnum*1000);
        res = splice("M",Mnum);
        int Cnum = 0;
        int Dnum = 0;
        if (num>=900){
            num = num -900;
            res += "CM";
        }else{
            if (num >= 400 && num <500){
                num -= 400;
                res += "CD";
            }else{
                Dnum = num / 500;
                num = num - (Dnum * 500);
                res += splice("D",Dnum);

            }
            Cnum = num / 100;
            num = num - (Cnum * 100);
            res += splice("C",Cnum);

        }
        int Lnum = 0;
        int Xnum = 0;
        if(num >=90){
            num -= 90;
            res += "XC";
        }else{
            if(num >= 40&& num <50){
                num -= 40;
                res += "XL";
            }else{
                Lnum = num / 50;
                num = num - (Lnum * 50);
                res += splice("L",Lnum);

            }
            Xnum = num / 10;
            num = num - (Xnum * 10);
            res += splice("X",Xnum);

        }
        int Vnum = 0;
        if (num >= 9){
            num -= 9;
            res += "IX";
        }else{
            if(num ==4){
                num -= 4;
                res += "IV";
            }else {
                Vnum = num / 5;
                num = num - (Vnum * 5);
                res += splice("V",Vnum)+splice("I",num);
            }
        }

        return res;

    }
    public String splice(String s , int num){
        String res = "";
        for (int i = 0; i < num; i++) {
            res += s;
        }
        return res;
    }

    /**
     * digits = "23"
     * 输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
     * @param args
     */
    public List<String> letterCombinations1(String digits) {
        //使用队列
        LinkedList<String> ans = new LinkedList<String>();
        if(digits.isEmpty()) return ans;
        String[] mapping = new String[] {"0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        ans.add("");
        for (int i = 0; i < digits.length(); i++) {
            int x = Character.getNumericValue(digits.charAt(i));
            while (ans.peek().length() == i){//查看队首元素
                String remove = ans.remove();//队首元素出队
                for (char c:mapping[x].toCharArray()) {
                    ans.add(remove + c);
                }
            }

        }
        return ans;

    }

    private static final String[] KEYS = { "", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz" };

    public List<String> letterCombinations(String digits) {
        if(digits.equals("")) {
            return new ArrayList<String>();
        }
        List<String> ret = new LinkedList<String>();
        combination("", digits, 0, ret);
        return ret;
    }

    private void combination(String prefix, String digits, int offset, List<String> ret) {
        //offset 代表在加哪个数字递归的出口
        if (offset == digits.length()) {
            ret.add(prefix);
            return;
        }
        String letters = KEYS[(digits.charAt(offset) - '0')];
        for (int i = 0; i < letters.length(); i++) {
            combination(prefix + letters.charAt(i), digits, offset + 1, ret);
        }
    }

    /**
     * 给定一个包含 n 个整数的数组 nums 和一个目标值 target，判断 nums 中是否存在四个元素 a，b，c 和 d
     * ，使得 a + b + c + d 的值与 target 相等？找出所有满足条件且不重复的四元组。
     *
     * 注意：答案中不可以包含重复的四元组。
     *
     * @param args
     */
    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> list=new ArrayList<>();
        Arrays.sort(nums);
        int n=nums.length;
        for(int i=0;i<n-3;i++){
            if(i>0&&nums[i]==nums[i-1]) continue;
            //如果 target-nums[i] 比3*nums[n-1]大，可以继续
            if(target-nums[i]>3*nums[n-1]) continue;
            //如果 target-nums[i] 比3*nums[n+1]小，可以继续
            if(target-nums[i]<3*nums[i+1]) break;
            for(int j=i+1;j<n-2;j++){
                if(j>i+1&&nums[j]==nums[j-1]) continue;
                int t=target-nums[i]-nums[j];
                if(t>2*nums[n-1]) continue;
                if(t<2*nums[j+1]){
                    break;
                }
                int k=j+1;int m=n-1;
                while(k<m){
                    if(nums[k]+nums[m]<t){
                        k++;
                    }else if(nums[k]+nums[m]>t){
                        m--;
                    }else{
                        List<Integer> list1=new ArrayList<>();
                        list1.add(nums[i]);
                        list1.add(nums[j]);
                        list1.add(nums[k]);
                        list1.add(nums[m]);
                        list.add(list1);
                        while(k<m-1&&nums[k]==nums[k+1]){
                            k++;
                        }
                        while(m>k+1&&nums[m]==nums[m-1]){
                            m--;
                        }
                        k++;
                        m--;
                    }
                }
            }
        }
        return list;
    }

    /**
     * 给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
     *
     * 进阶：你能尝试使用一趟扫描实现吗？
     * @param args
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        //好像使用双指针方法能行
        ListNode left = head;
        int i=1;
        ListNode right = head;
        while (i<n){
            right = right.next;
            i++;
        }
        ListNode pre_left = left;
        boolean b = true;
        while (right.next != null){
            right = right.next;
            pre_left = left;
            left = left.next;
            b = false;
        }
        pre_left.next = left.next;
        return b ? pre_left.next : head;
    }

    public ListNode removeNthFromEnd1(ListNode head, int n) {
        /*
        利用空间换时间，便利一次链表，将每个结点存入数组中，就知道那个是倒数地n个了
         */
        ListNode temp = head;
        List<ListNode> list = new ArrayList<>();
        list.add(head);
        int i = 1;
        while (temp.next != null){
            temp = temp.next;
            list.add(temp);
            i++;
        }
        if (i == 1) {
            return null;
        }
        int remove = i - n;
        if (remove == 0) {
            return head.next;
        }
        ListNode r = list.get(remove - 1);
        r.next = r.next.next;
        return head;

    }

    /**
     * [1,2,3,4,5]
     * @param args
     */
    public ListNode buildNode(int [] nums){
        ListNode head = new ListNode(nums[0]);
        ListNode temp = head;
        for (int i = 1; i < nums.length; i++) {
            temp.next = new ListNode(nums[i]);
            temp = temp.next;
        }
        return head;
    }

    /**
     *  匹配括号
     * @param args
     */
    public boolean isValid(String s) {
        //利用栈的特性
        Stack<Character> stack = new Stack<>();
        //遇见左括号就入栈，遇见右括号出栈
        String left = "({[";
        int b =0;
        for (int i = 0; i < s.length(); i++) {
            if (left.indexOf(s.charAt(i)) != -1){
                stack.add(s.charAt(i));
                b=1;
            }else {
                b=0;
                if (i==0){
                    return false;
                }
                if(stack == null || stack.isEmpty()){
                    continue;
                }
                Character character = stack.peek();
                switch (s.charAt(i)){
                    case ')' :
                        if (character-'(' !=0){
                            return false;
                        }else{
                            stack.pop();
                            b=1;
                            break;
                        }
                    case '}' :
                        if (character-'{' !=0){
                            return false;
                        }else{
                            stack.pop();
                            b=1;
                            break;
                        }
                    case ']' :
                        if (character-'[' !=0){
                            return false;
                        }else{
                            stack.pop();
                            b=1;
                            break;
                        }
                }

            }
        }
        return stack.isEmpty()&&b==1;

    }

    /**
     * 将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。
     * 递归
     * @param args
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode temp = new ListNode(0);
        help (temp,l1, l2);
        return temp.next;
    }
    public ListNode help (ListNode temp,ListNode l1, ListNode l2){
        if(l1 != null && l2 != null){
            if (l1.val<l2.val){
                temp.next =  l1;
                return help (temp.next,l1.next, l2);
            }else{
                temp.next =  l2;
                return help (temp.next,l1, l2.next);
            }
        }
        if (l1 == null){
            temp.next =  l2;
            return temp;
        }
        if (l2 == null){
            temp.next =  l1;
            return temp;
        }
        return null;
    }

    /**
     * 迭代
     * @param l1
     * @param l2
     * @return
     */
    public ListNode mergeTwoLists1(ListNode l1, ListNode l2) {
        ListNode head = new ListNode(0);
        ListNode temp = head;
        while (l1 !=null && l2 != null){
            if (l1.val > l2.val){
                temp.next = l2;
                l2 = l2.next;
            }else{
                temp.next = l1;
                l1 = l1.next;
            }
            temp = temp.next;
        }
        if (l1 == null && l2 == null){
            return null;
        }
        if(l1 == null){
            temp.next = l2;
        }
        if (l2 == null){
            temp.next = l1;
        }
        return head.next;
    }

    /**
     * 给你一个链表数组，每个链表都已经按升序排列。
     *
     * 请你将所有链表合并到一个升序链表中，返回合并后的链表
     * @param args
     */
    public ListNode mergeKLists(ListNode[] lists) {
        //重新定义优先队列的比较器
        Comparator<ListNode> cmp;
        cmp = new Comparator<ListNode>() {
            @Override
            public int compare(ListNode o1, ListNode o2) {
                return o1.val-o2.val;
            }
        };

        //建立队列
        Queue<ListNode> listNodesQueue = new PriorityQueue<>(cmp);
        for (ListNode listNode : lists){
            if (listNode != null)
                listNodesQueue.add(listNode);
        }
        ListNode head = new ListNode(0);
        ListNode point = head;
        while (!listNodesQueue.isEmpty()){
            //出队列
            point.next = listNodesQueue.poll();
            point = point.next;
            //判断当前链表是否为空，不为空就将新元素入队
            ListNode next = point.next;
            if(next!=null){
                listNodesQueue.add(next);
            }
        }
        return head.next;
    }
    //两两合并
    public ListNode mergeKLists1(int[] lists) {
        if (lists.length == 1){
          //  return lists[0];
        }
        if (lists.length == 0){
            return null;
        }
        int interval = 1;
        int j = 1;
        while(interval<lists.length){
            //System.out.println(lists.length);
            for (int i = 0; i + interval< lists.length; i=i+interval*2) {
                //lists[i]=mergeTwoLists(lists[i],lists[i+interval]);
                System.out.println("第"+j+"遍=========================");
                System.out.println(i);
                System.out.println(i+interval);
                j++;
            }
            interval*=2;
        }

        return null;
    }

    /**
     * 给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。
     *
     * 你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
     * @param args
     */
    public ListNode swapPairs(ListNode head) {
        ListNode h = new ListNode(0);
        h.next = head;
        ListNode temp = h;

        while (temp.next != null&& temp.next.next != null){
            ListNode swap1 = temp.next;
            ListNode swap2 = temp.next.next;
            temp.next = swap2;
            swap1.next = swap2.next;
            swap2.next = swap1;
            temp = swap1;
        }
        return h.next;
    }

    /**
     * 给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。
     *
     * k 是一个正整数，它的值小于或等于链表的长度。
     *
     * 如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
     *
     * 进阶：
     *
     *     你可以设计一个只使用常数额外空间的算法来解决此问题吗？
     *     你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。
     *
     * @param args
     */
    public ListNode reverseKGroup(ListNode head, int k) {
        if(k<3){
           return swapPairs(head);
        }
        //找到所有相隔K的节点的结合
        Map<Integer,ListNode> map = new HashMap<>();
        //维护这个缓存表
        ListNode knode = head;
        for (int i = 1; i < k; i++) {
            knode = knode.next;
        }
        int i = 1;
        //问题是，总是有重复的，所以到哪里停止非常关键
        while (knode !=  null){
            map.put(i,knode);
            knode = knode.next;
            i++;
        }
        //然后从第一个位置进行交换
        int j = 1;
        ListNode h = new ListNode(0);
        h.next = head;
        ListNode temp = h;
        while (!map.isEmpty()){
            ListNode swap1 = h.next;
            ListNode swap2 = map.get(j);
            temp.next = swap2;
            swap1.next = swap2.next;
            swap2.next = swap1;
            temp = swap1;
            map.remove(j);
            j++;
        }
        return temp.next;

    }

    /**
     * 给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。
     *
     * 不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。
     * 也就是说不能使用新数组
     * 要求删除重复元素，实际上就是将不重复的元素移到数组的左侧。
     *
     * @param args
     */
    public int removeDuplicates(int[] nums) {
        if (nums.length == 0) return 0;
        int i = 0;
        for (int j = 1; j < nums.length; j++) {
            if (nums[j] != nums[i]) {
                i++;
                nums[i] = nums[j];
            }
        }
        return i + 1;
    }

    /**
     * 给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。
     *
     * 不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。
     *
     * 元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
     * @param args
     */
    public int removeElement(int[] nums, int val) {
        int i = 0;int j = nums.length-1;
        for (; i < nums.length ;) {
            if (nums[i] == val&&i<j){
                int temp = nums[j];
                nums[j] = nums[i];
                nums[i] = temp;
                j--;
            }else
                i++;
        }
        int z = nums.length-1;
        for (; z >=0; z--) {
            if (nums[z] != val){
               return z+1;
            }
        }
        return 0;
    }

    /**
     * 给你两个字符串 haystack 和 needle ，请你在 haystack 字符串中找出 needle
     * 字符串出现的第一个位置（下标从 0 开始）。如果不存在，则返回  -1 。
     *
     * @param args
     */
    public int strStr(String haystack, String needle) {
        if ( "".equals(needle)){
            return 0;
        }
        int index = -1;
        for (int i = 0; i < haystack.length(); i++) {
            if (haystack.charAt(i) == needle.charAt(0)){
                if(index == -1){
                    index = i;
                }else{
                    break;
                }
                int k = i;
                for (int j = 0; j < needle.length(); j++) {

                    if (k+1>=haystack.length()&&k+1<needle.length()){
                        return -1;
                    }
                    if (haystack.charAt(k++) != needle.charAt(j)){
                        index = -1;
                        break;
                    }

                }
            }

        }
        return index;
    }

    /**
     * 给定两个整数，被除数 dividend 和除数 divisor。将两数相除，要求不使用乘法、除法和 mod 运算符。（模运算）
     *
     * 返回被除数 dividend 除以除数 divisor 得到的商。
     *
     * 整数除法的结果应当截去（truncate）其小数部分，例如：truncate(8.345) = 8 以及 truncate(-2.7335) = -2
     * @param args
     */
    public int divide(int dividend, int divisor) {
        //移位运算吗
        //难道进行减法的运算吗，dividend-divisor
        //将负数变正数是不行了，超时了
        boolean a = false;
        if ((dividend <0 && divisor >0) || (dividend >0 && divisor <0) ){
            a = true;//dividend 是负数，但是divisor不是
        }
        if(dividend <= Integer.MIN_VALUE && divisor == 1){
            return Integer.MIN_VALUE;
        }else if (dividend <= Integer.MIN_VALUE && divisor == -1){
            return Integer.MAX_VALUE;
        }
        Long dividend_long = Math.abs(Long.valueOf(dividend));
        Long divisor_long = Math.abs(Long.valueOf(divisor));

        int truncate = 0;
        //这里需要优化
        while (dividend_long >= divisor_long){
            dividend_long = dividend_long - divisor_long;
            truncate++;
        }
        if (truncate >= 2147483647){
            truncate = 2147483647;
        }
        if (a){
            truncate = truncate - (truncate+truncate);
        }
        return truncate;
    }

    /**
     * 给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，
     * 返回它将会被按顺序插入的位置。
     *
     * 你可以假设数组中无重复元素。
     *
     * @param args
     */
    public int searchInsert(int[] nums, int target) {
        //二分法
        int left = 0;int right = nums.length-1;
        while (left<=right){
            int mid = (right+left)/2;
            if (nums[mid] == target){
                return mid;
            }else if (nums[mid] < target){
                left = mid + 1;
            }else{
                right = mid -1;
            }
        }
        return left;


    }

    /**
     * 给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
     * @param args
     */
    public int maxSubArray(int[] nums) {
        if(nums.length == 1){
            return nums[0];
        }
        int max = nums[0];
        for (int i = 0; i < nums.length; i++) {
            //要想和最大，至少正数要多
        }
        return 0;
    }

    /**
     * 存在一个按升序排列的链表，给你这个链表的头节点 head ，请你删除所有重复的元素，使每个元素 只出现一次 。
     *
     * 返回同样按升序排列的结果链表。
     * @param args
     */
    public ListNode deleteDuplicates(ListNode head) {
        //保存头结点信息
        ListNode root = new ListNode(0);
        root.next= head;
        ListNode temp = head;
        while (head != null && temp.next != null ){
            ListNode l = temp.next;
            if (temp.val == l.val){
                temp.next = l.next;
                l.next = null;
            }else{
                temp = l;
            }
        }
        return root.next;
    }
    public ListNode deleteDuplicates1(ListNode head) {
        if (head == null) return  null;
        if (head.next == null) return head;
        if (head.val == head.next.val){
            head.next =  head.next.next;
            head = deleteDuplicates1(head);
        }else{
            head.next = deleteDuplicates1(head.next);
        }
        return head;
    }

    /**
     * 给定一个按照升序排列的整数数组 nums，和一个目标值 target。
     * 找出给定目标值在数组中的开始位置和结束位置。
     *
     * 如果数组中不存在目标值 target，返回 [-1, -1]。
     * 你可以设计并实现时间复杂度为 O(log n) 的算法解决此问题吗？
     *
     * @param args
     */
    public int[] searchRange(int[] nums, int target) {
        int start = 0;
        int end = nums.length - 1;
        int[] ans = { -1, -1 };
        if (nums.length == 0) {
            return ans;
        }
        while (start <= end) {
            int mid = (start + end) / 2;
            if (target == nums[mid]) {
                end = mid - 1;
            } else if (target < nums[mid]) {
                end = mid - 1;
            } else {
                start = mid + 1;
            }
        }
        //考虑 tartget 是否存在，判断我们要找的值是否等于 target 并且是否越界
        if (start == nums.length || nums[ start ] != target) {
            return ans;
        } else {
            ans[0] = start;
        }
        ans[0] = start;
        start = 0;
        end = nums.length - 1;
        while (start <= end) {
            int mid = (start + end) / 2;
            if (target == nums[mid]) {
                start = mid + 1;
            } else if (target < nums[mid]) {
                end = mid - 1;
            } else {
                start = mid + 1;
            }
        }
        ans[1] = end;
        return ans;
    }
    /**=============================排序算法.start================================*/
    //冒泡排序
    public int[] bubble(int []nums){
        for (int i = 0; i < nums.length; i++) {
            for (int j = i+1; j < nums.length; j++) {
                if (nums[i]>nums[j]){
                    int temp = nums[i];
                    nums[i] = nums[j];
                    nums[j] = temp;
                }
            }
        }
        return nums;
    }
    //希尔排序
    public int[] shell(int[] nums){
        //确定初始值
        int h =1;
        while (h<nums.length/2){
            h = h*2-1;
        }
        while (h>=1){
            for (int i = h; i < nums.length; i++) {
                for (int j = i; j < nums.length; j+=h) {
                    if (nums[i] > nums[j]){
                        int temp = nums[i];
                        nums[i] = nums[j];
                        nums[j] = temp;
                    }
                }
            }
            h = h/2;
        }
        return nums;
    }
    //插入排序--和冒泡反过来
    public int[] insert(int []nums){
        for (int i = 1; i < nums.length; i++) {
            for (int j = i-1; j > 0; j--) {
                if (nums[i]<nums[j]){
                    int temp = nums[i];
                    nums[i] = nums[j];
                    nums[j] = temp;
                }
            }
        }
        return nums;
    }
    //选择排序
    public int[] select(int[] nums){
        for (int i = 0; i < nums.length-2; i++) {
            int minIndex = i;
            for (int j = i+1; j < nums.length; j++) {
                if (nums[j] < nums[i]){
                    minIndex = j;
                }
            }
            int temp = nums[i];
            nums[i] = nums[minIndex];
            nums[minIndex] = temp;
        }
        return nums;
    }
    //堆排序
    /*public int[] heap(int[] nums){

    }*/
    //归并排序
    /**
     * 1.尽可能的一组数据拆分成两个元素相等的子组，并对每一个子组继续拆分，直到拆分后的每个子组的元素个数是
     * 1为止。
     *  2.将相邻的两个子组进行合并成一个有序的大组；
     *  3.不断的重复步骤2，直到最终只有一个组为止。
     */
    /**===============================排序算法。end====================================*/
    /**===============================深度优先遍历-------------------------------------*/
    /**
     * 100.相同二叉树
     * 给你两棵二叉树的根节点 p 和 q ，编写一个函数来检验这两棵树是否相同。
     * 如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。
     * @param args
     */
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) return true;
        if (p != null && q != null) {
            //判断自身结点是否相同
            if (p.val == q.val){
                return isSameTree(p.left,q.left)&&isSameTree(p.right,q.right);
            }else{
                return false;
            }
        }
        return false;
    }

    /**
     * 101:镜像二叉树
     * 给定一个二叉树，检查它是否是镜像对称的
     * @param args
     */
    public boolean isSymmetric3(TreeNode root) {
        if (root == null) return false;
        return help(root.left,root.right);
    }
    public boolean help(TreeNode p,TreeNode q){
        if (p == null && q == null) return true;
        if (p == null || q == null) return false;
        if (p.val == q.val){
            return help(p.left,q.right)&&help(p.right,q.left);
        }
        return false;
    }
    //迭代
    public boolean isSymmetric4(TreeNode root) {
        if (root == null) return false;
        Queue<TreeNode> queue1 = new LinkedList<>();
        Queue<TreeNode> queue2 = new LinkedList<>();
        if (root.left!=null && root.right!= null){
            queue1.add(root.left);
            queue2.add(root.right);
        }else if (root.left==null && root.right== null){
            return true;
        }else{
            return false;
        }
        while (!queue1.isEmpty() && !queue2.isEmpty()){
            TreeNode node1 = queue1.poll();
            TreeNode node2 = queue2.poll();
            if (node1.val == node2.val){
                if (node1.left != null && node2.right!= null){
                    queue1.add(node1.left);
                    queue2.add(node2.right);
                }else{
                    if(node1.left == null && node2.right == null){

                    }else{
                        return false;
                    }
                }
                if (node1.right != null && node2.left!= null){
                    queue1.add(node1.right);
                    queue2.add(node2.left);
                }else{
                    if(node1.right == null && node2.left == null){

                    }else{
                        return false;
                    }
                }
            }else{
                return false;
            }
        }
        return true;
    }

    /**
     * 102:给定一个二叉树，找出其最大深度。
     *
     * 二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
     * @param args
     */

    public int maxDepth1(TreeNode root) {
        if (root == null) return 0;
        int left = help(root.left,1);
        int right = help(root.right,1);
        return Math.max(left,right);
    }
    public int help (TreeNode node,int depth){
        if (node == null) return depth;
        if (node.left == null && node.right == null) return depth;
        return Math.max(help(node.right,depth),help(node.left,depth))+1;
    }

    /**108:
     * 给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 高度平衡 二叉搜索树。
     *
     * 高度平衡 二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过 1 」的二叉树。
     *
     *
     * 二叉搜索树:左子树小于根节点小于右子节点
     * 若任意节点的左子树不空，则左子树上所有节点的值均小于它的根节点的值；
     * 若任意节点的右子树不空，则右子树上所有节点的值均大于它的根节点的值；
     * 任意节点的左、右子树也分别为二叉查找树；
     * 没有键值相等的节点
     * @param args
     */
    public TreeNode sortedArrayToBST(int[] nums) {
        //平衡二叉树
        //类比，已知前序遍历和中序遍历怎么还原二叉树呢？就是找到根节点，由顶向下
        //回到此题，nums的中点当做二叉树的根是不是有行呢
        if (nums.length == 0) return null;
        return help(nums,0,nums.length);
    }
    public TreeNode help(int []nums,int left,int right){
        if (left >= right) {
            return null;
        }
        int mid = (left + right) >>> 1;//无符号右移
        TreeNode root = new TreeNode(nums[mid]);
        root.left = help(nums,left,mid);
        root.right = help(nums,mid+1,right);
        return root;
    }

    /**
     * 99.恢复二叉树
     * 给你二叉搜索树的根节点 root ，该树中的两个节点被错误地交换。请在不改变其结构的情况下，恢复这棵树。
     *
     * 进阶：使用 O(n) 空间复杂度的解法很容易实现。你能想出一个只使用常数空间的解决方案吗
     *
     * @param args
     */
    public void recoverTree(TreeNode root) {
    }
    /**
     * 102二叉树的层序遍历
     * 给你一个二叉树，请你返回其按 层序遍历 得到的节点值。 （即逐层地，从左到右访问所有节点）
     */
    public List<List<Integer>> levelOrder2(TreeNode root) {
        if (root == null) return new ArrayList<>();
        List<List<Integer>> list = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()){
            List<Integer> node_list = new ArrayList<>();
            Queue<TreeNode> temp = new LinkedList<>();
            for (TreeNode node : queue) {
                node_list.add(node.val);
                TreeNode left = node.left;
                TreeNode right = node.right;
                if (left != null) temp.add(left);
                if (right != null) temp.add(right);
            }
            queue = temp;
            list.add(node_list);
        }
        return list;
    }
    /**107
     * 给定一个二叉树，返回其节点值自底向上的层序遍历。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）
     */
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        if (root ==null) return new ArrayList<>();
        Map<Integer,List<Integer>> map = new HashMap<>();
        List<List<Integer>> res = new ArrayList<>();
        int i=0;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()){
            Queue<TreeNode> queue1 = new LinkedList<>();
            List<Integer> list = new ArrayList<>();
            for (int j = 0; j <queue.size() ; j++) {
                TreeNode temp = queue.peek();
                list.add(temp.val);
                if (temp.left != null) queue1.add(temp.left);
                if (temp.right != null) queue1.add(temp.right);
            }
            queue = queue1;
            map.put(i,list);
            i++;
        }
        while (i>=0){
            res.add(map.get(i));
            i--;
        }
        return res;
    }

    /**110.平衡二叉树
     * 给定一个二叉树，判断它是否是高度平衡的二叉树。
     * 本题中，一棵高度平衡二叉树定义为：
     *     一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1 。
     * @param args
     * 执行用时：1 ms, 在所有 Java 提交中击败了99.99% 的用户
     * 内存消耗：38.7 MB, 在所有 Java 提交中击败了24.70% 的用户
     */
    public boolean isBalanced1(TreeNode root) {
        if(root == null){
            return true;
        }
        return Math.abs(getDepth4(root.left)-getDepth4(root.right))>1 && isBalanced1(root.left)&&isBalanced1(root.right);
    }
    public int getDepth4(TreeNode root){
        if (root == null){
            return 0;
        }

        int right = getDepth4(root.right)+1;
        int left = getDepth4(root.left)+1;
        return Math.max(left,right);
    }

    /**111.树的最小深度
     * 给定一个二叉树，找出其最小深度。
     *
     * 最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
     *
     * 说明：叶子节点是指没有子节点的节点。
     * [5,0,-4,-1,-6,-9,null,7,null,1,3,null,0,null,9,null,null,6,0,null,-7,null,null,null,null,null,null,-4,null,1,null,null,-4]
     * @param args
     */
    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return minDepthHelper(root);

    }

    private int minDepthHelper(TreeNode root) {
        //到达叶子节点就返回 1
        if (root.left == null && root.right == null) {
            return 1;
        }
        //左孩子为空，只考虑右孩子的方向
        if (root.left == null) {
            return minDepthHelper(root.right) + 1;
        }
        //右孩子为空，只考虑左孩子的方向
        if (root.right == null) {
            return minDepthHelper(root.left) + 1;
        }
        //既有左孩子又有右孩子，那么就选一个较小的
        return Math.min(minDepthHelper(root.left), minDepthHelper(root.right)) + 1;
    }

    /**
     * 116.填充每个结点的右侧字指针
     */
    public Node connect(Node root) {
        if (root == null){
            return null;
        }
        Node pre = root;
        Node cur = null;
        Node start = pre;
        while (pre.left != null){
            if(cur == null){
                pre.left.next = pre.right;

                pre = start.left;
                cur = start.right;
                start = pre;
            }else{
                pre.left.next = pre.right;
                pre.right.next = cur.left;
                pre = pre.next;
                cur = cur.next;
            }
        }
        return root;
    }
    /**
     * 118.杨辉三角
     * 给定一个非负整数 numRows，生成杨辉三角的前 numRows 行。
     */
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> ans = new ArrayList<>();
        for (int i = 0; i < numRows; i++) {
            List<Integer> sub = new ArrayList<>();
            for (int j = 0; j <= i; j++) {
                if (j == 0 || j == i) {
                    sub.add(1);
                } else {
                    List<Integer> last = ans.get(i - 1);
                    sub.add(last.get(j - 1) + last.get(j));
                }

            }
            ans.add(sub);
        }
        return ans;
    }


    /**
     * 121
     * 对于数组 1 6 2 8，代表股票每天的价格。
     *
     * 定义一下转换规则，当前天的价格减去前一天的价格，第一天由于没有前一天，规定为 0，用来代表不操作。
     *
     * 数组就转换为 0 6-1 2-6 8-2，也就是 0 5 -4 6。现在的数组的含义就变成了股票相对于前一天的变化了。
     *
     * 现在我们只需要找出连续的和最大是多少就可以了，也就是变成了 53 题。
     *
     * 连续的和比如对应第 3 到 第 6 天加起来的和，那对应的买入卖出其实就是第 2 天买入，第 6 天卖出。
     *
     * 换句话讲，买入卖出和连续的和形成了互相映射，所以问题转换成功。
     * @param prices
     * @return
     */
    public int maxProfit(int[] prices) {
        int maxProfit = 0;
        int buy = 0;
        int sell = 0;
        for (; sell < prices.length; sell++) {
            //当前价格更小了，更新 buy
            if (prices[sell] < prices[buy]) {
                buy = sell;
            } else {
                maxProfit = Math.max(maxProfit, prices[sell] - prices[buy]);

            }
        }
        return maxProfit;
    }

    /**
     * 给你二叉树的根节点 root 和一个表示目标和的整数 targetSum ，
     * 判断该树中是否存在 根节点到叶子节点 的路径，这条路径上所有节点值相加等于目标和 targetSum 。
     *
     * 叶子节点 是指没有子节点的节点。
     * @param args
     */
    public boolean hasPathSum(TreeNode root, int targetSum) {
        return false;
    }

    /**
     * 122.买卖股票的最佳时机
     * 给定一个数组 prices ，其中 prices[i] 是一支给定股票第 i 天的价格。
     *
     * 设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
     *
     * 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）
     * @param args
     */
    public int maxProfit1(int[] prices) {
        //只要当前天比前一天的高就可以买入
        int sell = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i]>prices[i-1]){
                sell += prices[i]-prices[i-1];
            }
        }
        return sell;

    }

    /**123.
      *  给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。
      *  设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。
     */
    public int maxProfit3(int[] prices) {
        int [] difference = new int[prices.length-1];
        for (int i = 1; i < prices.length; i++) {
            difference[i-1] = prices[i] - prices[i-1];
        }
        int max=0;
        Comparator<Integer> comparator = new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o2-o1;
            }
        };
        Queue<Integer> queue = new PriorityQueue<>(comparator);
       // Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < difference.length; i++) {
            int aim = i+1<difference.length ? difference[i+1] :0;
            if (difference[i]>0 || (max +difference[i]>0&&max <aim )){
                max +=difference[i];
            }else{
                queue.add(max);
                max = 0;
            }
            if(i== difference.length-1){
                queue.add(max);
            }

        }
        int j = 1;
        int res =0;
        while (!queue.isEmpty()&&j<=2){
            int temp =queue.poll();
            res += temp;
            j++;
        }
        return res;

    }

    /**
     * 344
     * 编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 char[] 的形式给出。
     *
     * 不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。
     *
     * 你可以假设数组中的所有字符都是 ASCII 码表中的可打印字符。
     * @param args
     */
    public void reverseString(char[] s) {
        int left=0;int right = s.length-1;
        while(left<right){
            char temp = s[left];
            s[left] = s[right];
            s[right] = temp;
            left++;
            right--;
        }
    }

    /**136.只出现一次的数字
     * 给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
     *
     * 说明：
     * 你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？
     * @param args
     */
    public int singleNumber(int[] nums) {
        System.out.println(nums.hashCode());
        //没有开辟新的内存空间，排好序的数组
        Arrays.sort(nums);
        System.out.println(nums.hashCode());
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            if(i%2 == 0){//偶数位相加
                sum += nums[i];
            }else{//奇数位相减
                sum -= nums[i];
            }
        }
        return sum;

    }

    /**
     *
     * @param args
     */
    public int search(int[] arr, int target) {
        int left = 0;int right = arr.length;
        while (left<=right){
            int mid = left + (right-left)/2;
            if(arr[mid] == target){
                return mid;
                //
            }else if(arr[mid] < target){
                //看mid以左是否是升序的
                if(arr[right] <= target){
                    right = mid-1;
                }else {
                    left = mid+1;
                }


            }else{
                if (arr[left] > target){
                    left= mid+1;
                }else {
                    right = mid-1;
                }
            }
        }
        return left;
    }

    /**
     * 4.寻找2个正序的数组的下标
     * @param nums1
     * @param nums2
     * @return
     */
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int [] nums3 = new int[nums1.length+nums2.length];
        //第一个，原数组，第二个从什么位置起，新数组，初始位置，复制多长
        System.arraycopy(nums1,0,nums3,0,nums1.length);
        System.arraycopy(nums2,0,nums3,nums1.length,nums2.length);
        Arrays.sort(nums3);
        int a = 0;
        int b = nums3.length-1;
        int mid = a+(b-a)/2;
        if (nums3.length%2==0){
            return Double.valueOf(nums3[mid] + nums3[mid+1])/2;
        }else{
            return nums3[mid];
        }
    }

    /**
     * 找到子数组中连续的有相同数量0和1
     * @param args
     */
    public int findMaxLength(int[] nums) {
       /* int max = 0;
        Map<Integer,Integer> map = new HashMap<>();
        int counter = 0;
        map.put(counter,-1);
        for (int i = 0; i < nums.length; i++) {
            if (nums[i]==1){
                counter++;
            }else{
                counter--;
            }
            if (map.containsKey(counter)){
                int de = map.get(counter);
                max = Math.max(max,i-de);
            }else{
                map.put(counter,i);
            }
        }
        return max;*/

        int[] arr = new int[nums.length * 2 + 1];
        Arrays.fill(arr, -2);
        arr[nums.length] = -1;
        int maxlen = 0, count = 0;
        for (int i = 0; i < nums.length; i++) {
            count = count + (nums[i] == 0 ? -1 : 1);
            if (arr[count + nums.length] >= -1) {
                maxlen = Math.max(maxlen, i - arr[count + nums.length]);
            } else {
                arr[count + nums.length] = i;
            }
        }
        return maxlen;
    }

    /**
     * 给你一个整数数组 nums 和一个整数 k ，编写一个函数来判断该数组是否含有同时满足下述条件的连续子数组：
     *
     *     子数组大小 至少为 2 ，且
     *     子数组元素总和为 k 的倍数。
     *
     * 如果存在，返回 true ；否则，返回 false 。
     *
     * 如果存在一个整数 n ，令整数 x 符合 x = n * k ，则称 x 是 k 的一个倍数
     * @param args
     */
    public boolean checkSubarraySum(int[] nums, int k) {
        int m = nums.length;
        if (m < 2) {
            return false;
        }
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        map.put(0, -1);
        int remainder = 0;
        for (int i = 0; i < m; i++) {
            remainder = (remainder + nums[i]) % k;
            if (map.containsKey(remainder)) {
                int prevIndex = map.get(remainder);
                if (i - prevIndex >= 2) {
                    return true;
                }
            } else {
                map.put(remainder, i);
            }
        }
        return false;

    }

    /**
     * 128.最长连续子序列
     * @param nums
     * @return
     */
    public int longestConsecutive(int[] nums) {
        HashSet<Integer> set = new HashSet<>();
        for (int i = 0; i < nums.length; i++) {
            set.add(nums[i]);
        }
        int max = 0;
        for (int i = 0; i < nums.length; i++) {
            int num = nums[i];
            if (!set.contains(num-1)){
                int count=0;
                while (set.contains(num)){
                    count++;
                    num++;
                }
                max = Math.max(max,count);
            }
        }
        return max;
    }

    public int longestConsecutive1(int[] nums) {
        if (nums.length == 0) return 0;

        // 首次遍历，与邻居结盟
        UnionFind uf = new UnionFind(nums);
        for (int v : nums)
            uf.union(v, v + 1); // uf.union() 结盟

        // 二次遍历，记录领队距离
        int max = 1;
        for (int v : nums)
            max = Math.max(max, uf.find(v) - v + 1); // uf.find() 查找领队
        return max;
    }

    class UnionFind {
        private int count;
        private Map<Integer, Integer> parent; // (curr, leader)

        UnionFind(int[] arr) {
            parent = new HashMap<>();
            for (int v : arr)
                parent.put(v, v); // 初始时，各自为战，自己是自己的领队

            count = parent.size(); // 而非 arr.length，因可能存在同 key 的情况
            // 感谢 [@icdd](/u/icdd/) 同学的指正
        }

        // 结盟
        void union(int p, int q) {
            // 不只是 p 与 q 结盟，而是整个 p 所在队伍 与 q 所在队伍结盟
            // 结盟需各领队出面，而不是小弟出面
            Integer rootP = find(p), rootQ = find(q);
            if (rootP == rootQ) return;
            if (rootP == null || rootQ == null) return;

            // 结盟
            parent.put(rootP, rootQ); // 谁大听谁
            // 应取 max，而本题已明确 p < q 才可这么写
            // 当前写法有损封装性，算法题可不纠结

            count--;
        }

        // 查找领队
        Integer find(int p) {
            if (!parent.containsKey(p))
                return null;

            // 递归向上找领队
            int root = p;
            while (root != parent.get(root))
                root = parent.get(root);

            // 路径压缩：扁平化管理，避免日后找领队层级过深
            while (p != parent.get(p)) {
                int curr = p;
                p = parent.get(p);
                parent.put(curr, root);
            }

            return root;
        }
    }

    /**130.
     * 给你一个 m x n 的矩阵 board ，由若干字符 'X' 和 'O' ，找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。
     */
    public void solve(char[][] board) {

    }


    /**
     * 单例模式，双重检索机制
     */
    class singleton{
        public volatile  singleton singleton = null;
        public singleton newinstansce(){
            if (singleton == null)
                synchronized (singleton.class){
                    if (singleton == null){
                        singleton = new singleton();
                    }
                }
            return singleton;
       }
    }

    /**
     * 有效的数独
     * @param board
     * @return
     */
    public boolean isValidSudoku(char[][] board) {
        Set seen = new HashSet();
        for (int i=0; i<9; ++i) {
            for (int j=0; j<9; ++j) {
                if (board[i][j] != '.') {
                    String b = "(" + board[i][j] + ")";
                    if (!seen.add(b + i) || !seen.add(j + b) || !seen.add(i/3 + b + j/3))
                        return false;
                }
            }
        }
        return true;
    }
/** ==============================================二分专题 。start-----------------------------------------------------------------**/
    /**
     *  38.外观数列
     * @param args
     */
    public String countAndSay(int n) {
        if (n==1){
            return "1";
        }
        String last = countAndSay(n-1);
        //输出下一个字串
        return getNextString(last);
    }
    public String getNextString(String last){
        if ("".equals(last)){
            return "";
        }
        //计算重复的数量
        int num = getRepeatNum(last);
        return num + "" + last.charAt(0) + getNextString(last.substring(num));

    }
    //得到字符 string[0] 的重复个数，例如 "111221" 返回 3
    private int getRepeatNum(String string) {
        int count = 1;
        char same = string.charAt(0);
        for (int i = 1; i < string.length(); i++) {
            if (same == string.charAt(i)) {
                count++;
            } else {
                break;
            }
        }
        return count;
    }

    /**
     * 已知存在一个按非降序排列的整数数组 nums ，数组中的值不必互不相同。
     *
     * 在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转 ，
     * 使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。
     * 例如， [0,1,2,4,4,4,5,6,6,7] 在下标 5 处经旋转后可能变为 [4,5,6,6,7,0,1,2,4,4] 。
     *
     * 给你 旋转后 的数组 nums 和一个整数 target ，请你编写一个函数来判断给定的目标值是否存在于数组中。如果 nums
     * 中存在这个目标值 target ，则返回 true ，否则返回 false
     *
     * @param args
     */
    public boolean search2(int[] nums, int target) {
        int n = nums.length;
        if (n == 0) {
            return false;
        }
        if (n == 1) {
            return nums[0] == target;
        }
        int l = 0, r = n - 1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (nums[mid] == target) {
                return true;
            }
            if (nums[l] == nums[mid] && nums[mid] == nums[r]) {
                ++l;
                --r;
            } else if (nums[l] <= nums[mid]) {
                if (nums[l] <= target && target < nums[mid]) {
                    r = mid - 1;
                } else {
                    l = mid + 1;
                }
            } else {
                if (nums[mid] < target && target <= nums[n - 1]) {
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            }
        }
        return false;
    }

    public int findMin(int[] nums) {
        int left = 0;int right = nums.length-1;
        while (left<right){
            int mid = left +(right-left)/2;
            if (nums[left]>nums[mid]){
                if (nums[right]>nums[mid]){
                    right = mid;
                }else{
                    left = mid;

                }
            }else if (nums[left]<nums[mid]){
                if (nums[left]<nums[right]){
                   break;
                }else{
                    left = mid;
                }
            }else {
                if (nums[left]>nums[right]){
                    return nums[right];
                }
                break;
            }
        }
        return nums[left];
    }

    /**162.寻找峰值
     * 峰值元素是指其值大于左右相邻值的元素。
     *
     * 给你一个输入数组 nums，找到峰值元素并返回其索引。数组可能包含多个峰值，在这种情况下，返回 任何一个峰值 所在位置即可。
     *
     * 你可以假设 nums[-1] = nums[n] = -∞
     *  = [1,2,3,1]
     * @param nums
     * @return
     */
    public int findPeakElement(int[] nums) {
        int left=0;int right =nums.length-1;
        while(left<=right){
            int mid = left+(right-left)/2;
            if (nums[mid]>nums[mid+1]){
                left = mid+1;
            }else {
                right = mid;
            }
        }
        return left;
    }

    /**209.长度最小的子数组
     * 给定一个含有 n 个正整数的数组和一个正整数 target 。
     *
     * 找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。
     * @param target
     * @param nums
     * @return
     */
    public int minSubArrayLen(int target, int[] nums) {
        //滑动窗口
        int left =0;int right =0;
        int aims =0;
        int min = Integer.MAX_VALUE;
        while(right<nums.length-1){
            aims = aims + nums[right];
            right++;
            while (aims >= target){
                min = Math.min(min,right-left);
                aims -= nums[left];
                left = left++;
            }
        }
        return min==Integer.MAX_VALUE ? 0:min;
    }

    /**
     *猜数字游戏的规则如下：
     *
     *     每轮游戏，我都会从 1 到 n 随机选择一个数字。 请你猜选出的是哪个数字。
     *     如果你猜错了，我会告诉你，你猜测的数字比我选出的数字是大了还是小了。
     *
     * 你可以通过调用一个预先定义好的接口 int guess(int num) 来获取猜测结果，返回值一共有 3 种可能的情况（-1，1 或 0）：
     *
     *     -1：我选出的数字比你猜的数字小 pick < num
     *     1：我选出的数字比你猜的数字大 pick > num
     *     0：我选出的数字和你猜的数字一样。恭喜！你猜对了！pick == num
     *
     * 返回我选出的数字。
     *
     * @param matrix
     * @param target
     * @return
     */
    public int guessNumber(int n) {
        int left = 1;int right = n;
        while (left <= right){
            int mid = left+(right-left)/2;
            if (mid == 6){
                return mid;
            }else if (mid < 6){
                left = mid+1;
            }else{
                right = mid-1;
            }
        }
        return left;
    }

    /**
     * 给定两个数组，编写一个函数来计算它们的交集
     * @param args
     */
    public int[] intersection(int[] nums1, int[] nums2) {
        HashSet<Integer> set = new HashSet();
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < nums1.length; i++) {
            set.add(nums1[i]);
        }
        for (int i = 0; i < nums2.length; i++) {
            if (set.contains(nums2[i])&&!list.contains(nums2[i])){
                list.add(nums2[i]);
            }
        }
        int[] ints = new int[list.size()];
        for (int i = 0; i <list.size() ; i++) {
            ints[i] = list.get(i);
        }
        return ints;
    }

    /**367.完全平方数
     * 给定一个 正整数 num ，编写一个函数，如果 num 是一个完全平方数，则返回 true ，否则返回 false 。
     *
     * 进阶：不要 使用任何内置的库函数，如  sqrt
     * @param args
     */
    public boolean isPerfectSquare(int num) {
        int left=1;int right= num;
        while(left<=right){
            int mid = (right+left)>>>1;
           // int mid_num = mid*mid;越界
            int t = num / mid;
            if (t == mid){
                return true;
            }else if (t < mid){
                right = mid-1;
            }else{
                left = mid+1;
            }
        }
        return false;
    }

    /**392.判断子序列
     *字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，"ace"是"abcde"的一个子序列，而"aec"不是）。
     *
     * 进阶：
     * 如果有大量输入的 S，称作 S1, S2, ... , Sk 其中 k >= 10亿，你需要依次检查它们是否为 T 的子序列。在这种情况下，你会怎样改变代码？
     * @param args
     */
    public boolean isSubsequence(String s, String t) {
        //双指针法
        int a = 0;
        int b = 0;
        while (b<t.length()&&a<s.length()){
            if (s.charAt(a) == t.charAt(b)){
                a++;
                b++;
            }else{
                b++;
            }
        }
        return a == s.length();
    }

    /**
     * 441.排列硬币
     * 你总共有 n 枚硬币，你需要将它们摆成一个阶梯形状，第 k 行就必须正好有 k 枚硬币。
     *
     * 给定一个数字 n，找出可形成完整阶梯行的总行数。
     *
     * n 是一个非负整数，并且在32位有符号整型的范围内。
     * @return
     */
    public int arrangeCoins(int n) {
        //等差数列求和
        //Sn = k(a(1)+a(K))/2
        //等比数列求和
        //(a(1)-a(n)*q)/1-q
        int left = 0;int right = n;
        while(left<=right){
            int mid = (right+left)>>>1;
            long Sn = mid*(1+mid) /2;
            if (Sn == n){
                return mid;
            }else if(Sn < n){
                left = mid + 1;
            }else {
                right = mid - 1;
            }
        }
        return left-1;
    }

    /**
     * 给你一个排序后的字符列表 letters ，列表中只包含小写英文字母。另给出一个目标字母 target，请你寻找在这一有序列表里比目标字母大的最小字母。
     *
     * 在比较时，字母是依序循环出现的。举个例子：
     *
     *     如果目标字母 target = 'z' 并且字符列表为 letters = ['a', 'b']，则答案返回 'a'
     */
    public char nextGreatestLetter(char[] letters, char target) {
        int left = 0;int right = letters.length-1;
        while(left <= right){
            int mid = left + (right - left) / 2;
            if (letters[mid] == target){
                //防止重复的，就后移
                if(mid+1 <= letters.length-1)  left = mid +1;
                if(mid == letters.length-1) return letters[0];
            }else if (letters[mid] > target){
                right = mid -1;
            }else {
                left = mid +1;
            }
        }
        //有可能该数组中就没有这个target，所以left也就越界
        return left <= letters.length-1 ? letters[left] : letters[0];
    }

    /**
     * 704.二分查找
     * 给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1
     */
    public int search1(int[] nums, int target) {
        int left = 0; int right = nums.length-1;
        while(left <= right){
            int mid = (right+left)>>>1;
            if (nums[mid] == target){
                return mid;
            }else if (nums[mid] > target){
                left = mid + 1;
            }else{
                right = mid - 1;
            }
        }
        //因为是查找吗？查到了就在上面返回了，没查到就返回-1;
        return -1;
    }

    /**852.
     * 符合下列属性的数组 arr 称为 山脉数组 ：
     *
     *     arr.length >= 3
     *     存在 i（0 < i < arr.length - 1）使得：
     *         arr[0] < arr[1] < ... arr[i-1] < arr[i]
     *         arr[i] > arr[i+1] > ... > arr[arr.length - 1]
     *
     * 给你由整数组成的山脉数组 arr ，返回任何满足 arr[0] < arr[1] < ... arr[i - 1] < arr[i] > arr[i + 1] > ... > arr[arr.length - 1] 的下标 i 。
     * 只有一个峰顶
     */
    public int peakIndexInMountainArray(int[] arr) {
        int left = 0; int right = arr.length-1;
        //思路：比较中值，判断mid+1和mid 的值大小，
        while(left<right){
            int mid = (right+left)>>>1;
            if (arr[mid] < arr[mid+1]){
                left = mid+1;
            }else{
                right = mid;
            }
        }
        return left;
    }

    /**1351
     * 给你一个 m * n 的矩阵 grid，矩阵中的元素无论是按行还是按列，都以非递增顺序排列。
     * 请你统计并返回 grid 中 负数 的数目。
     */
    public int countNegatives(int[][] grid) {
        //非递增就是递减和相等
        int n = grid.length;
        int nums = 0;
        for (int i = 0; i < n; i++) {
            int [] row = grid[i];
            int left = 0;int right = row.length-1;
            int pos = row.length;
            while (left<=right){
                int mid = (right+left)>>>1;
                if (row[mid]>=0){
                    left = mid+1;
                }else{
                    pos = mid;
                    right = mid -1;
                }
            }
            nums = nums + row.length - pos;
        }
        return nums;
    }

    /**230.二叉搜索树中的第k小的元素
     * 给定一个二叉搜索树的根节点 root ，和一个整数 k ，请你设计一个算法查找其中第 k 个最小元素（从 1 开始计数）。
     */
    public int K =0;
    public int kthSmallest(TreeNode root, int k) {
        //二叉搜索树最重要的性质就是中序遍历是一个升序数组
        //中序遍历：左，中，右
        K = k;
        Stack<TreeNode> keys = new Stack<>();
        midErgodic(root,keys);

        return keys.peek().val;
    }
    //使用中序遍历，获取指定树x中所有的键，并存放到key中
    TreeNode temp = null;
    private void midErgodic(TreeNode x,Stack<TreeNode> keys){
        if (x==null){
            return;
        }
        if (keys.size() == K){
            return;
        }
        //先递归，把左子树中的键放到keys中
        if (x.left!=null){
            midErgodic(x.left,keys);
        }
        //把当前结点x的键放到keys中
        if (keys.size() <= K) temp = x;
        //在递归，把右子树中的键放到keys中
        if(x.right!=null){
            midErgodic(x.right,keys);
        }

    }

    /**
     * 436.寻找右区间
     * 给你一个区间数组 intervals ，其中 intervals[i] = [starti, endi] ，且每个 starti 都 不同 。
     *
     * 区间 i 的 右侧区间 可以记作区间 j ，并满足 startj >= endi ，且 startj 最小化 。
     *
     * 返回一个由每个区间 i 的 右侧区间 的最小起始位置组成的数组。如果某个区间 i 不存在对应的 右侧区间 ，则下标 i 处的值设为 -1 。
     */
    public int[] findRightInterval(int[][] intervals) {
        return new int[1];
    }








    /** ==============================================二分专题 。end-----------------------------------------------------------------**/

    /** ==============================================动态规划。start-------------------------------------------------------------------*/


    /** ==============================================动态规划 end-------------------------------------------------------------------*/

    /** ==============================================链表 。start-----------------------------------------------------------------------*/
    /**203.移出链表中的元素
     * 给你一个链表的头节点 head 和一个整数 val ，请你删除链表中所有满足 Node.val == val 的节点，并返回 新的头节点
     */
    public ListNode removeElements(ListNode head, int val) {
        ListNode root = new ListNode(0);
        root.next = head;
        ListNode pre = null;
        while (head != null){
            if (head.val != val){
                pre = head;
            }else{
                if (pre != null){
                    pre.next = head.next;
                }else{
                    root.next = head.next;
                }

            }
            head = head.next;
        }
        return root.next;
    }

    /**876.链表的中间结点
     *
     * 给定一个头结点为 head 的非空单链表，返回链表的中间结点。
     *
     * 如果有两个中间结点，则返回第二个中间结点。
     */
    public ListNode middleNode(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    /**1290.二进制
     * 给你一个单链表的引用结点 head。链表中每个结点的值不是 0 就是 1。已知此链表是一个整数数字的二进制表示形式。
     *
     * 请你返回该链表所表示数字的 十进制值
     */
    public int getDecimalValue(ListNode head) {
        int N = 0;
        ListNode head_temp = head;
        while (head != null){
            head = head.next;
            N++;
        }
        int num = 0;
        while (head_temp != null){
            if (head_temp.val == 1){
                num += Math.pow(2,N-1);
            }
            N--;
            head_temp = head_temp.next;
        }
        return num;
    }

    /** 中等*/
    /**61.旋转链表
     * 给你一个链表的头节点 head ，旋转链表，将链表每个节点向右移动 k 个位置
     */
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null || k == 0) {
            return head;
        }
        int len = 0;
        ListNode h = head;
        ListNode tail = null;
        //求出链表长度，保存尾指针
        while (h != null) {
            h = h.next;
            len++;
            if (h != null) {
                tail = h;
            }
        }
        //求出需要整体移动多少个节点
        int n = k % len;
        if (n == 0) {
            return head;
        }

        //利用快慢指针找出倒数 n + 1 个节点的指针，用 slow 保存
        ListNode fast = head;
        while (n >= 0) {
            fast = fast.next;
            n--;
        }
        ListNode slow = head;
        while (fast != null) {
            slow = slow.next;
            fast = fast.next;
        }
        //尾指针指向头结点
        tail.next = head;
        //头指针更新为倒数第 n 个节点
        head = slow.next;
        //尾指针置为 null
        slow.next = null;
        return head;
    }

    /**92.反转链表II
     *
     * 给你单链表的头指针 head 和两个整数 left 和 right ，其中 left <= right 。请你反转从位置 left 到位置 right 的链表节点，返回 反转后的链表
     */
    public ListNode reverseBetween(ListNode head, int left, int right) {

        //记录两个节点，left前一个和right后一个
        //依次将left到right的节点入栈
        if(left == right) return head;
        ListNode root1 = new ListNode(-1);
        ListNode root = head;
        Stack<ListNode> stack = new Stack<ListNode>();
        ListNode pre_left = null;
        ListNode aft_right = null;
        int i = 1;
        while(head != null && i <= right){
            if (i == left-1){
                pre_left = head;
            }else if(i >= left &&i <= right){
                aft_right = head.next;
                stack.add(head);
            }
            head = head.next;
            i++;
        }
        int n = stack.size();
        for (int j = 0; j < n; j++) {
            if(pre_left != null){
                pre_left.next = stack.pop();
                pre_left = pre_left.next;
                if (j == n-1){
                    pre_left.next = aft_right;
                }
            }else{
                if (root1.next == null) {
                    root = root1;
                }
                root1.next = stack.pop();
                root1 = root1.next;
                if (j == n-1){
                    root1.next = aft_right;
                    if(aft_right == null){
                        return root;
                    }else{
                        return root.next;
                    }
                }
            }

        }
        return root;
    }

    /**82.删除排序链表中的重复的元素II
     * 存在一个按升序排列的链表，给你这个链表的头节点 head ，请你删除链表中所有存在数字重复情况的节点，只保留原始链表中 没有重复出现 的数字。
     *  1 2 3 3 4 4 5
     * 返回同样按升序排列的结果链表
     */
    public ListNode deleteDuplicates2(ListNode head) {
        ListNode pre = new ListNode(0);
        ListNode dummy = pre;
        pre.next = head;
        ListNode cur = head;
        while(cur!=null && cur.next!=null){
            boolean equal = false;
            //cur 和 cur.next 相等，cur 不停后移
            while(cur.next!=null && cur.val == cur.next.val){
                cur = cur.next;
                equal = true;
            }
            //发生了相等的情况
            // pre.next 直接指向 cur.next 删除所有重复数字
            if(equal){
                pre.next = cur.next;
                equal = false;
                //没有发生相等的情况
                //pre 移到 cur 的地方
            }else{
                pre = cur;
            }
            //cur 后移
            cur = cur.next;
        }
        return dummy.next;
    }

    /**
     * 编写代码，移除未排序链表中的重复节点。保留最开始出现的节点
     */
    public ListNode removeDuplicateNodes(ListNode head) {
        if (head == null) return head;
        HashMap<Integer,Integer> map = new HashMap<>();
        ListNode root = head;
        //一个一个的比较，如果前一个小于后一个就要，如果相等或者后一个小，就不要
        ListNode pre = null;
        while (head != null){
           if (map.containsKey(head.val)){
               pre.next = head.next;
             //  pre = pre.next;  前一个结点位置不变
           }else{
               pre = head;
               map.put(head.val,1);
           }

           head = head.next;
        }
        return root;
    }



    /**
     * 面试题 02.06. 回文链表
     * 编写一个函数，检查输入的链表是否是回文的。
     *
     * 利用hashMap保存值和索引，当下次遇见这个key，计算map里的索引和i相加是否等于n-1，并且i要大于索引的中间
     *
     * 进阶：时间复杂度n和空间复杂度是1
     */
    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null) return true;
        ListNode root = head;
        int n = 0;
        while (head != null){
            head = head.next;
            n++;
        }

        HashMap<Integer, Integer> hashMap = new HashMap<>();

        int i = 0;
        while (root != null){
            if (hashMap.containsKey(root.val)){
                if (hashMap.get(root.val)+i != n-1 && i>(n-1)/2){
                    return false;
                }else if(hashMap.get(root.val)+i == n-1){
                    hashMap.remove(root.val);
                }
            }else{
                hashMap.put(root.val,i);
            }
            root = root.next;
            i++;
        }
        if (hashMap.size() > 1){
            return false;
        }
        if(hashMap.size() == 0){
            return true;
        }
        Integer [] values = new Integer[1];
        values = hashMap.values().toArray( values);
        return values[0] == (n-1)/2;
    }

    /**
     * 给定两个（单向）链表，判定它们是否相交并返回交点。请注意相交的定义基于节点的引用，而不是基于节点的值。
     * 换句话说，如果一个链表的第k个节点与另一个链表的第j个节点是同一节点（引用完全相同），则这两个链表相交。
     *
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode a = headA;
        ListNode b = headB;
        int a1 = 0;
        int b1 = 0;
        while (a != null && b != null){
            if (a == b){
                return a;
            }else{
                if (a.next == null && a1 == 0){
                    a = headB;
                    a1++;
                }else {
                    a = a.next;
                }
                if (b.next == null && b1 == 0){
                    b = headA;
                    b1++;
                }else {
                    b = b.next;
                }
            }
        }
        return null;
    }

    /**面试题 02.03. 删除中间节点
     * 若链表中的某个节点，既不是链表头节点，也不是链表尾节点，则称其为该链表的「中间节点」。
     *
     * 假定已知链表的某一个中间节点，请实现一种算法，将该节点从链表中删除。
     *
     * 例如，传入节点 c（位于单向链表 a->b->c->d->e->f 中），将其删除后，剩余链表为 a->b->d->e->f
     */
    public void deleteNode(ListNode node) {
        node.val = node.next.val;
        node.next = node.next.next;
    }

    /**109. 有序链表转换二叉搜索树
     * 给定一个单链表，其中的元素按升序排序，将其转换为高度平衡的二叉搜索树。
     *
     * 本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1
     *
     */
    public TreeNode sortedListToBST(ListNode head) {
        return null;
    }

    /**445. 两数相加 II
     * 给你两个 非空 链表来代表两个非负整数。数字最高位位于链表开始位置。它们的每个节点只存储一位数字。将这两数相加会返回一个新的链表。
     *
     * 你可以假设除了数字 0 之外，这两个数字都不会以零开头
     *
     * 进阶：
     *
     * 如果输入链表不能修改该如何处理？换句话说，你不能对列表中的节点进行翻转。
     */
    public ListNode addTwoNumbers2(ListNode l1, ListNode l2) {
        Stack<ListNode> s1 = new Stack<>();
        Stack<ListNode> s2 = new Stack<>();
        while (l1 != null){
            s1.add(l1);
            l1 = l1.next;
        }
        while (l2 != null){
            s2.add(l2);
            l2 = l2.next;
        }
        int flag = 0;
       // ListNode node = new ListNode(-1);
        ListNode temp = null;
        while (!s1.isEmpty() && !s2.isEmpty()){
            ListNode t1 = s1.pop();
            ListNode t2 = s2.pop();
            int i = t1.val+t2.val+flag;
            if (flag == 0){
                if (i>9){
                    i -= 10;
                    flag =1;
                }
            }else {
                if (i>9){
                    i -= 10;
                }else{
                    flag =0;
                }
            }
            ListNode n = new ListNode(i);
            n.next = temp;
            temp = n;
        }
        while (!s1.isEmpty()){
            ListNode t1 = s1.pop();
            int i = t1.val+flag;
            if (flag == 0){
                if (i>9){
                    i -= 10;
                    flag =1;
                }
            }else {
                if (i>9){
                    i -= 10;
                }else{
                    flag =0;
                }
            }
            ListNode n = new ListNode(i);
            n.next = temp;
            temp = n;
        }
        while (!s2.isEmpty()){
            ListNode t2 = s2.pop();
            int i = t2.val+flag;
            if (flag == 0){
                if (i>9){
                    i -= 10;
                    flag =1;
                }
            }else {
                if (i>9){
                    i -= 10;
                }else{
                    flag =0;
                }
            }
            ListNode n = new ListNode(i);
            n.next = temp;
            temp = n;
        }
        if (flag == 1){
            ListNode n = new ListNode(1);
            n.next = temp;
            temp = n;
        }
        return temp;

    }
    /**142. 环形链表 II
     * 给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
     *
     * 为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意，
     * pos 仅仅是用于标识环的情况，并不会作为参数传递到函数中。
     *
     * 说明：不允许修改给定的链表。
     *
     * 进阶：
     *
     *     你是否可以使用 O(1) 空间解决此题？
     *
     */
    public ListNode detectCycle(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        int i = 0;
        while ( fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast){
                break;
            }
        }
        slow = head;
        while (fast != null){
            slow = slow.next;
            fast = fast.next;
            if (slow == fast) return slow;
        }
        return null;
    }

    /**
     * 给定一个单链表 L：L0→L1→…→Ln-1→Ln ，
     * 将其重新排列后变为： L0→Ln→L1→Ln-1→L2→Ln-2→…
     *
     * 你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
     *
     * 示例 2:
     *
     * 给定链表 1->2->3->4->5, 重新排列为 1->5->2->4->3.
     */
    public void reorderList(ListNode head) {
        ListNode later = head;
        Stack<ListNode> stack = new Stack<>();
        while(later != null){
            stack.add(later);
            later = later.next;
        }
        ListNode cur = head;
        ListNode pre = head;
        int i =1;
        while (cur != null){
            if(i != 1){
                ListNode node1 = stack.peek();
                if(node1 != cur){
                    ListNode node3 = stack.pop();
                    pre.next = node3;
                    node1.next = cur;
                }else{
                    break;
                }

            }
            pre = cur;
            cur = cur.next;

            if(i >= stack.size()) break;
            i++;
        }
        ListNode node2 = stack.pop();
        node2.next = null;

    }
    /**148.排序链表
     * 给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表
     */
    public ListNode sortList(ListNode head) {
        int n = 0;
        ListNode cur = head;
        while(cur != null){
            cur = cur.next;
            n++;
        }
        //冒泡排序，超时了，只能使用归并排序
        for (int i = 0; i < n; i++) {
            ListNode node = head;
            while(node != null && node.next != null){
                if(node.val > node.next.val){
                    int temp = node.val;
                    node.val = node.next.val;
                    node.next.val = temp;
                }
                node = node.next;
            }
        }
        return head;
    }
    /**
     * 给你链表的头节点 head 和一个整数 k 。
     *
     * 交换 链表正数第 k 个节点和倒数第 k 个节点的值后，返回链表的头节点（链表 从 1 开始索引）。给你链表的头节点 head 和一个整数 k 。
     *
     */

        public ListNode swapNodes(ListNode head, int k) {
            ListNode dummy = new ListNode(0);
            dummy.next = head;// 因为头结点可能会发生交换，所以要构造一个哑结点
            ListNode pre1 = dummy;// pre1指向第k个节点的前一个节点
            ListNode left = dummy.next;// 第k个节点
            ListNode pre2 = dummy;// pre2指向倒数第k个节点的前一个节点
            ListNode right = dummy.next;// 倒数第k个节点
            for(int i = 1; i < k; i++){
                pre1 = pre1.next;
                left = left.next;
            }
            ListNode cur = left;
            ListNode temp = left.next;// 第k个节点的后一个节点
            while(cur.next != null){
                pre2 = pre2.next;
                right = right.next;
                cur = cur.next;
            }
            if(right == pre1){// 特殊情况，倒数第k个节点在第k个节点的左侧
                right.next = temp;
                left.next = right;
                pre2.next = left;}
            else{
                left.next = right.next;
                if(pre2 == left){right.next = left;}// 特殊情况，第k个节点在倒数第k个节点的右侧
                else{
                    pre2.next = left;
                    right.next = temp;
                }
                pre1.next = right;
            }
            return dummy.next;
        }





    /** ==============================================链表 。 end-----------------------------------------------------------------------*/
    /** ==============================================堆 。 start------------------------------------------------------------------------*/

    /** ==============================================堆 。 end--------------------------------------------------------------------------*/
    /** ===============================================哈希表 start ----------------------------------------------------------*/
    /**202.快乐数
     * 编写一个算法来判断一个数 n 是不是快乐数。
     *
     * 「快乐数」定义为：
     *
     *     对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和。
     *     然后重复这个过程直到这个数变为 1，也可能是 无限循环 但始终变不到 1。
     *     如果 可以变为  1，那么这个数就是快乐数。
     *
     * 如果 n 是快乐数就返回 true ；不是，则返回 false 。
     * @param args
     */
    public boolean isHappy(int n) {
        HashSet<Integer> set = new HashSet<>();
        set.add(n);
        while (true){
            int next = getNext(n);
            if (next == 1){
                return true;
            }
            if (!set.add(next)){
                return false;
            }
            n = next;
        }

    }
    //计算各个位的平方和
    private int getNext(int n) {
        int next = 0;
        while (n > 0) {
            int t = n % 10;
            next += t * t;
            n /= 10;
        }
        return next;
    }

    /** ===============================================哈希表 end ----------------------------------------------------------*/
    /** ===============================================树。start -----------------------------------------------------------*/
    /**112.路径总和
     * 给你二叉树的根节点 root 和一个表示目标和的整数 targetSum ，判断该树中是否存在 根节点到叶子节点 的路径，这条路径上所有节点值相加等于目标和 targetSum 。
     *
     * 叶子节点 是指没有子节点的节点。
     */
    public boolean hasPathSum1(TreeNode root, int targetSum) {
        if (root ==  null) return targetSum==0;
        return dfs(root,targetSum-root.val);
    }
    private boolean dfs(TreeNode node,int target){
        if (node.left == null && node.right == null ){
            return target-node.val==0;
        }

        if (node.left != null&& node.right==null) return dfs(node.left,target-node.left.val);
        if (node.right != null&& node.left == null) return dfs(node.right,target-node.right.val);
        return dfs(node.left,target-node.left.val) || dfs(node.right,target-node.right.val);

    }

    public boolean hasPathSum2(TreeNode root, int targetSum) {
        if (root == null){
            return false;
        }
        TreeNode cur = root;
        int curSum = 0;
        Stack<TreeNode> stack = new Stack<>();
        Stack<Integer> stack_int = new Stack<>();
        //由底向上，由左到右，递归的迭代描述
        while (cur != null || !stack.isEmpty()){
            //结点不是空就一致入栈，此时
            while (cur != null){
                stack.add(cur);
                curSum += cur.val;
                stack_int.add(curSum);
                cur = cur.left;
            }
            //结点为空出栈，
            cur =stack.pop();
            curSum = stack_int.pop();
            //先看是否时叶子结点
            if(curSum == targetSum && cur.left==null&&cur.right==null){
                return true;
            }
            //再看看有没有有结点，
            cur = cur.right;
        }
        return false;
    }

    /**257.二叉树的所有路径和
     * 给定一个二叉树，返回所有从根节点到叶子节点的路径。
     *输出: ["1->2->5", "1->3"]
     * 说明: 叶子节点是指没有子节点的节点。
     */
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> list = new ArrayList<>();
        dfs(root,list,"");
        return list;
    }
    private void dfs(TreeNode node, List<String> list,String s){
        if (node.left == null && node.right == null){
            s += ""+node.val;
            list.add(s);
            return;
        }
        if (node.left != null) dfs(node.left,list,s+""+node.val+"->");
        if (node.right != null) dfs(node.right,list,s+""+node.val+"->");
    }

    /**563.二叉树的坡度
     * 给定一个二叉树，计算 整个树 的坡度 。
     *
     * 一个树的 节点的坡度 定义即为，该节点左子树的节点之和和右子树节点之和的 差的绝对值 。如果没有左子树的话，左子树的节点之和为 0 ；
     * 没有右子树的话也是一样。空结点的坡度是 0 。
     * 整个树 的坡度就是其所有节点的坡度之和。
     */
    int slope=0;
    public int findTilt(TreeNode root) {
        dfs(root);
        return slope;
    }
    //获得左右子树结点和的差值
    private int dfs(TreeNode node){
        if (node == null){
            return 0;
        }
        int left = dfs(node.left);
        int right = dfs(node.right);
        slope += Math.abs(left-right);
        return left+right+root.val;
    }

    /**690.员工的重要性
     *给定一个保存员工信息的数据结构，它包含了员工 唯一的 id ，重要度 和 直系下属的 id 。
     *
     * 比如，员工 1 是员工 2 的领导，员工 2 是员工 3 的领导。他们相应的重要度为 15 , 10 , 5 。那么员工 1 的数据结构是 [1, 15, [2]] ，员工 2的 数据结构是 [2, 10, [3]] ，员工 3 的数据结构是 [3, 5, []] 。注意虽然员工 3 也是员工 1 的一个下属，但是由于 并不是直系 下属，因此没有体现在员工 1 的数据结构中。
     *
     * 现在输入一个公司的所有员工信息，以及单个员工 id ，返回这个员工和他所有下属的重要度之和。
     */
    public int getImportance(List<Employee> employees, int id) {
        HashMap<Object, Integer> hashMap = new HashMap<>();
        int importance = 0;
        List<Integer> subordinates = new ArrayList<>();
        //subordinates.add(id);
        for (Employee em:employees) {
            if(id == em.id ){
                importance += em.importance;
                subordinates = em.subordinates;
            }
            if(subordinates.contains(em.id)){
                importance += em.importance;
                subordinates.remove(subordinates.indexOf(em.id));
                subordinates.addAll(em.subordinates);
            }
            hashMap.put(em.id,employees.indexOf(em));

        }
        if (!subordinates.isEmpty()){
            for (int i = 0; i < subordinates.size(); i++) {
                Employee em = employees.get(hashMap.get(subordinates.get(i)));
                importance += em.importance;
                subordinates.addAll(em.subordinates);
            }
        }
        return importance;
    }

    class Employee {
        public int id;
        public int importance;
        public List<Integer> subordinates;
    };

    /**559.N叉树的最大深度
     * 给定一个 N 叉树找到其最大深度，。
     *
     * 最大深度是指从根节点到最远叶子节点的最长路径上的节点总数。
     *
     * N 叉树输入按层序遍历序列化表示，每组子节点由空值分隔（请参见示例）。
     *
     */
    public int maxDepth(Node1 root) {
        if (root == null){
            return 0;
        }
        if (root.children == null){
            return 1;
        }
        int maxdepth = 0;
        for (int i = 0; i < root.children.size(); i++) {
            maxdepth = Math.max(maxDepth(root.children.get(i))+1,maxdepth);
        }
        return maxdepth;
    }

    class Node1 {
        public int val;
        public List<Node1> children;

        public Node1() {}

        public Node1(int _val) {
            val = _val;
        }

        public Node1(int _val, List<Node1> _children) {
            val = _val;
            children = _children;
        }
    };

    /**113.路径总和II
     * 给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。
     *
     * 叶子节点 是指没有子节点的节点。
     *
     */
   /* public List<List< >> pathSum(TreeNode root, int targetSum) {

    }
    private void dfs(TreeNode root,List<List<Integer>> list){
        if (root == null){
            return ;
        }

    }*/

    /**
     * 在二叉树中，根节点位于深度 0 处，每个深度为 k 的节点的子节点位于深度 k+1 处。
     *
     * 如果二叉树的两个节点深度相同，但 父节点不同 ，则它们是一对堂兄弟节点。
     *
     * 我们给出了具有唯一值的二叉树的根节点 root ，以及树中两个不同节点的值 x 和 y 。
     *
     * 只有与值 x 和 y 对应的节点是堂兄弟节点时，才返回 true 。否则，返回 false
     *
     */
    public boolean isCousins(TreeNode root, int x, int y) {
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while(queue.isEmpty()){
            Queue<TreeNode> temp = new LinkedList<>();
            for ( TreeNode node: queue) {
                if(node.left != null) temp.add(node.left);
                if (node.right != null) temp.add(node.right);

            }
        }
        return true;
    }
    private void dfs(){

    }


    /** ===============================================树。end -------------------------------------------------------------*/
    /**
     * 给定一个包括 n 个整数的数组 nums 和 一个目标值 target。找出 nums 中的三个整数，使得它们的和与 target 最接近。
     * 返回这三个数的和。假定每组输入只存在唯一答案。
     *
     * @param args
     */
    public int threeSumClosest(int[] nums, int target) {
        int min = Integer.MAX_VALUE;
        for (int i = 0; i < nums.length; i++) {
            nums[i] = nums[i] - target;
        }
        return -1;

    }

    /**
     * 全排列
     * @param nums
     * @return
     */
    public List<List<Integer>> permute(int[] nums) {
        if(nums.length == 0){
            return new ArrayList();
        }
        List<List<Integer>> list = new ArrayList<>();
        boolean [] booleans = new boolean[nums.length];

        int depth = 1;
        Stack<Integer> path = new Stack<>();

        backtrack(nums,list,path,depth,nums.length,booleans);
        return null;
    }
    public void backtrack(int[] nums,List<List<Integer>> list,Stack<Integer> path,int depth,int n,boolean []booleans){
        if(depth == n){
            list.add(new ArrayList<>(path));
            return;
        }
        for (int i = 0;i<n;i++){
            if(booleans[i]){
                continue;
            }
            path.push(nums[i]);
            booleans[i] = true;
            backtrack(nums,list,path,depth+1,n,booleans);
            path.pop();
            booleans[i] = false;
        }
    }

    public int longestPalindromeSubseq(String s) {
        int n = s.length();
        int [][] dp = new int [n][n];
        for(int i = 0;i<s.length();i++){
            for(int j = i+1;j<n;j++){
                if(dp[i] == dp[j]){
                    dp[i][j] = dp[i+1][j-1]+2;
                }else{
                    dp[i][j] = Math.max(dp[i+1][j],dp[i][j-1]);
                }

            }
        }
        return dp[0][n-1];
    }

    public List<String> generateParenthesis(int n) {
        List<String> list = new ArrayList<String>();
        char [] arr = new char[2*n];
        int pose = 0;
        generateParentHelpher(pose,arr,list);
        return list;
    }
    public void generateParentHelpher(int pos,char[] arr,List<String> list){
        if(arr.length == pos) {
            if (valite(arr)) {
                list.add(arr.toString());
            }
        }else{
            arr[pos] = '(';
            generateParentHelpher(pos+1,arr,list);
            arr[pos] = ')';
            generateParentHelpher(pos+1,arr,list);
        }

    }
    public boolean valite(char [] arr){
        int bootm = 0;
        for(char c : arr){
            if(c == '('){
                ++bootm;
            }else{
                --bootm;
            }
            if(bootm < 0){
                return false;
            }
        }
        return bootm == 0;
    }

    /**
     * 继续
     * @param args
     */

    public static void main(String[] args) {
        int [] a1 = new int[]{
                100,90

        };
        int [] b1 = new int[]{2};
        int [][] c = new int[][]{
                {3,2},{1,0}
        };
        int [][] c1 = new int[][]{
                {2,1},{1,3},{1,2},{1,2},{2,1},{1,1},{1,2},{2,2}
        };
        char [] t = new char[]{
                'c','f','j'
        };
        Integer [] f = new Integer[]{3,9,20,null,null,15,7};
        //String []a = new String []{"JgZkQBoFzW", "OOI Jhncw", "dHtFhkkXvGmbomYFsT", "hSrUyWaU"};
        //[0,1,2,3,4,5,6,7,9]["at", "", "", "reverse", "ball", "", "", "car", "", "", "dad", "", ""]
        //"ta"["JgZkQBoFzW", "OOI Jhncw", "dHtFhkkXvGmbomYFsT", "hSrUyWaU"]
        //"jgT ChqUFnkxyNdgfWxz" [1,1,2,3,3,4,4,8,8]
        //"ball"
        String []strs = new String[]{"flower"};
        TwoNum twoNum = new TwoNum();
        String pa = "aaa";
        String os = "tmmzuxt";
        twoNum.swapNodes(twoNum.buildNode(a1),2);
      //  System.out.println(twoNum.countNegatives(c));
        //  System.out.println('!'-0);

        /**
         * 目光所及之处皆是你，所念之处皆星河
         */
        //  System.out.println(list.contains(arrayList1));
        System.out.println("字符串下标："+"1234566".charAt(0));


        //  twoNum.zigzagLevelOrder(twoNum.constructTree(f));
        long currentTimeMillis = System.currentTimeMillis();
        int a=0;
        int times = 10000*10000;
        for (long i = 0; i < times; i++) {
            a=9999%1024;
        }
        long currentTimeMillis2 = System.currentTimeMillis();

        int b=0;
        for (long i = 0; i < times; i++) {
            b=9999&(1024-1);
        }

        long currentTimeMillis3 = System.currentTimeMillis();
        System.out.println(a+","+b);
        System.out.println("%: "+(currentTimeMillis2-currentTimeMillis));
        System.out.println("&: "+(currentTimeMillis3-currentTimeMillis2));
       // List<String> list = new ArrayList<>(Arrays.asList(strs));
       /* List<String> list = Arrays.stream(strs).collect(Collectors.toList());
        list.add("jsj");*/
        List<Integer> list = new ArrayList<>();
        for (int i = 1; i <= 10; ++i) {
            list.add(i);
        }
        list.removeIf(s -> s % 2 == 0); /* 删除list中的所有偶数 */
        System.out.println(list);

        Class<? extends TwoNum> aClass = twoNum.getClass();
        System.out.println("aClass.getClassLoader() = " + aClass.getClassLoader());
        Method[] methods = aClass.getMethods();
        System.out.println("methods.length = " + methods.length);
    }


}
