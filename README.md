# Comment2Votes

Torch implementation of recurrent network architecture that predicts the number of upvotes a reddit comment will get.

The architecture has an encode and decode stage. The encode stage takes in char values. The decode stage is prompted by an EOS tag (-1 in the current implementation) and outputs a number representing the predicted votes. 
