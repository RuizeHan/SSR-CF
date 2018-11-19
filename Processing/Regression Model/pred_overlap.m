function overlap = pred_overlap(model, response)

Y = bsxfun(@plus, response*model.Beta(1:end-1), model.Beta(end));
Y = bsxfun(@plus, Y*model.T_inv, model.mu);
overlap = Y;

end