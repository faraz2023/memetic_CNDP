# memetic_CNDP
Memetic Search for Identifying Critical Nodes in Sparse Graphs

Working toward implementing: https://arxiv.org/abs/1705.04119
# memetic_CNDP
Memetic Search for Identifying Critical Nodes in Sparse Graphs

Working toward implementing: https://arxiv.org/abs/1705.04119

# a bit about how the code works (as far as I know)
- MACNP_interface is where we are running the inference. 
We have to have a structure for the EXP_LIST directory, where we have:
## structure of the list, should be like MEMETIC_CNDP/EXP_LIST etc etc
EXP_LIST/
│
├── report_fp.csv
│
├── exp_label_1/
│   ├── G.el
│   ├── hr0-/
│   │   └── overall_sol.txt
│   └── hr08-/
│       └── overall_sol.txt
│
├── exp_label_2/
│   ├── G.el
│   ├── hr0-/
│   │   └── overall_sol.txt
│   └── hr08-/
│       └── overall_sol.txt
│
... (more experiment labels)

the report_fp.csv should be like this: 
exp_label
exp_label_1
exp_label_2

and so on.
~You need to have it like this.

## Next in the results dir:
You have to have a G_MACNP.txt{K}.res1 in the results/model dir. 
The structure of this is like so:

   Computational results:                                 
   K = 408
   Limit time = 40.000000
   best objective value = 2605.000000
   time to find the best objective value = 34.800000
   number of steps to find the objective value = 1055472
   number of generations to find the objective value = 364
   number of removed nodes = 408
   Best removed nodes:
Computational results:                                 
   K = # of nodes you can remove
   Limit time = # of seconds you allow the algo to run
   best objective value = I put a random number here, depending on other examples
   time to find the best objective value = 171.810000 
   number of steps to find the objective value = some big amount, view example
   number of generations to find the objective value = # some amount, not sure if its being used
   number of removed nodes = # at the end, the total # of nodes removed, this needs to be the same as the # of nodes you list below the "best removed nodes"
   Best removed nodes: 
in this single line, have all the removed nodes. I will post an example below

## Example of how G_MACNP.txt{K}.res1 should look like:
G_MACNP.txt408.res1:
   Computational results:                                 
   K = 408
   Limit time = 40.000000
   best objective value = 2605.000000
   time to find the best objective value = 34.800000
   number of steps to find the objective value = 1055472
   number of generations to find the objective value = 364
   number of removed nodes = 408
   Best removed nodes:
774 665 849 846 930 828 813 803 697 692 740 617 351 286 282 240 111 649 317 45 910 981 841 911 909 905 1011 69 896 876 767 882 763 214 628 616 545 721 801 743 98 758 36 946 790 719 928 718 717 644 642 633 621 733 583 875 625 716 615 575 568 567 378 725 570 358 273 878 527 421 231 965 150 498 804 1008 544 32 516 492 482 993 144 652 581 888 277 903 336 258 610 847 477 461 937 189 838 60 773 764 590 86 297 81 877 624 97 687 934 522 989 907 123 749 177 528 920 702 1019 480 446 428 414 691 536 440 269 90 427 467 955 412 389 108 884 787 23 860 815 738 213 559 514 402 419 386 246 823 706 137 314 292 839 709 650 112 588 126 255 237 418 802 770 730 368 309 982 318 119 31 39 300 238 198 104 339 554 1013 580 416 234 898 352 188 940 187 939 280 218 202 647 320 949 270 6 47 361 983 604 699 287 549 253 203 18 259 682 143 272 62 923 128 999 759 484 388 342 322 827 760 91 43 38 957 950 148 346 304 525 658 433 146 742 52 396 654 271 135 61 142 84 230 241 222 8 783 601 121 557 775 59 15 92 558 938 935 782 861 893 778 766 169 44 48 120 872 814 722 732 840 1 51 79 157 85 667 862 432 515 584 985 233 705 956 966 20 139 299 186 880 87 890 662 771 423 166 521 486 224 984 881 785 632 252 341 338 873 776 637 256 945 520 232 191 316 564 302 679 589 50 294 37 606 313 298 10 859 627 464 599 5 1001 833 219 1017 925 698 109 927 704 101 926 35 359 356 1002 12 349 623 385 620 851 543 980 465 481 664 729 100 1007 330 493 383 308 865 576 312 3 363 125 577 434 660 808 867 426 325 523 245 9 392 954 635 751 784 354 425 988 889 696 247 932 685 626 556 452 370 959 912 850 811 532 474 323 497 207 892 845   Computational results:                                 
   K = 345
   Limit time = 240.000000
   best objective value = 2613.000000
   time to find the best objective value = 171.810000
   number of steps to find the objective value = 4154987
   number of generations to find the objective value = 2626
   number of removed nodes = 408
   Best removed nodes:
774 665 849 846 930 828 813 803 697 692 740 617 351 286 282 240 111 649 317 45 910 981 841 911 909 905 1011 69 896 876 767 882 763 214 628 616 545 721 801 743 98 758 36 946 790 719 928 718 717 644 642 633 621 733 583 875 625 716 615 575 568 567 378 725 570 358 273 878 527 421 231 965 150 498 804 1008 544 32 516 492 482 993 144 652 581 888 277 903 336 258 610 847 477 461 937 189 838 60 773 764 590 86 297 81 877 624 97 687 934 522 989 907 123 749 177 528 920 702 1019 480 446 428 414 691 536 440 269 90 427 467 955 412 389 108 884 787 23 860 815 738 213 559 514 402 419 386 246 823 706 137 314 292 839 709 650 112 588 126 255 237 418 802 770 730 368 309 982 318 119 31 39 300 238 198 104 339 554 1013 580 416 234 898 352 188 940 187 939 280 218 202 647 320 949 270 6 47 361 983 604 699 287 549 253 203 18 259 682 143 272 62 923 128 999 759 484 388 342 322 827 760 91 43 38 957 950 148 346 304 525 658 433 146 742 52 396 654 271 135 61 142 84 230 241 222 8 783 601 121 557 775 59 15 92 558 938 935 782 861 893 778 766 169 44 48 120 872 814 722 732 840 1 51 79 157 85 667 862 432 515 584 985 233 705 956 966 20 139 299 186 880 87 890 662 771 423 166 521 486 224 984 881 785 632 252 341 338 873 776 637 256 945 520 232 191 316 564 302 679 589 50 294 37 606 313 298 10 859 627 464 599 5 1001 833 219 1017 925 698 109 927 704 101 926 35 359 356 1002 12 349 623 385 620 851 543 980 465 481 664 729 100 1007 330 493 383 308 865 576 312 3 363 125 577 434 660 808 867 426 325 523 245 9 392 954 635 751 784 354 425 988 889 696 247 932 685 626 556 452 370 959 912 850 811 532 474 323 497 207 892 845

I have pasted this right from one of the working files.
## IF you dont get "ssh permission denied"
chmod +x MACNP.exe
Run this, to all the MACNP.exe to run

## What should the hr0- and hr8- include? 
I didn't put anything in them, they were empty, I think they are used to calcualte other scores for gurobi connectivity and hybrid_08 connectivity respectively.

## results:
After this, the output of the MACNP should come in a text file called MACNP_sol.txt. This includes the nodes removed, not sure if they are in order they are removed.
the G_MACNO.txt file is the adjacency list for the graph. It simply takes the networkx graph and makes it into that, dont worry about it.

# 